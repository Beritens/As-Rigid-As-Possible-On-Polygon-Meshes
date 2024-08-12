// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "custom_arap.h"


#include <cassert>

#include <iostream>

template <typename Scalar>
using MatrixXX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

bool custom_arap_precomputation(
     const Eigen::MatrixXd& V,
     const Eigen::MatrixXi& F,
     int dim,
     const Eigen::VectorXi& b,
     igl::ARAPData& data)
{
  using namespace std;
  using namespace Eigen;
  // number of vertices
  const int n = V.rows();
  data.n = n;
  assert((b.size() == 0 || b.maxCoeff() < n) && "b out of bounds");
  assert((b.size() == 0 || b.minCoeff() >=0) && "b out of bounds");
  // remember b
  data.b = b;
  //assert(F.cols() == 3 && "For now only triangles");
  // dimension
  //const int dim = V.cols();
  assert((dim == 3 || dim ==2) && "dim should be 2 or 3");
  data.dim = dim;
  //assert(dim == 3 && "Only 3d supported");
  // Defaults
  data.f_ext = MatrixXd::Zero(n,data.dim);

  assert(data.dim <= V.cols() && "solve dim should be <= embedding");
  bool flat = (V.cols() - data.dim)==1;

  MatrixXX<double> plane_V;
  MatrixXX<int> plane_F;
  typedef SparseMatrix<double> SparseMatrixS;
  SparseMatrixS ref_map,ref_map_dim;
  if(flat)
  {
    igl::project_isometrically_to_plane(V,F,plane_V,plane_F,ref_map);
    igl::repdiag(ref_map,dim,ref_map_dim);
  }
  const MatrixXX<double>& ref_V = (flat?plane_V:V);
  const MatrixXX<int>& ref_F = (flat?plane_F:F);
  SparseMatrixS L;
  igl::cotmatrix(V,F,L);

  igl::ARAPEnergyType eff_energy = data.energy;
  if(eff_energy == igl::ARAP_ENERGY_TYPE_DEFAULT)
  {
    switch(F.cols())
    {
      case 3:
        if(data.dim == 3)
        {
          eff_energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
        }else
        {
          eff_energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
        }
        break;
      case 4:
        eff_energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
        break;
      default:
        assert(false);
    }
  }


  // Get covariance scatter matrix, when applied collects the covariance
  // matrices used to fit rotations to during optimization
  covariance_scatter_matrix(ref_V,ref_F,eff_energy,data.CSM);
  if(flat)
  {
    data.CSM = (data.CSM * ref_map_dim.transpose()).eval();
  }
  assert(data.CSM.cols() == V.rows()*data.dim);

  // Get group sum scatter matrix, when applied sums all entries of the same
  // group according to G
  SparseMatrix<double> G_sum;
  if(data.G.size() == 0)
  {
    if(eff_energy == igl::ARAP_ENERGY_TYPE_ELEMENTS)
    {
      igl::speye(F.rows(),G_sum);
    }else
    {
      igl::speye(n,G_sum);
    }
  }else
  {
    // groups are defined per vertex, convert to per face using mode
    if(eff_energy == igl::ARAP_ENERGY_TYPE_ELEMENTS)
    {
      Eigen::Matrix<int,Eigen::Dynamic,1> GG;
      MatrixXi GF(F.rows(),F.cols());
      for(int j = 0;j<F.cols();j++)
      {
        GF.col(j) = data.G(F.col(j));
      }
      igl::mode<int>(GF,2,GG);
      data.G=GG;
    }
    //printf("group_sum_matrix()\n");
    igl::group_sum_matrix(data.G,G_sum);
  }
  SparseMatrix<double> G_sum_dim;
  igl::repdiag(G_sum,data.dim,G_sum_dim);
  assert(G_sum_dim.cols() == data.CSM.rows());
  data.CSM = (G_sum_dim * data.CSM).eval();


  arap_rhs(ref_V,ref_F,data.dim,eff_energy,data.K);
  if(flat)
  {
    data.K = (ref_map_dim * data.K).eval();
  }
  assert(data.K.rows() == data.n*data.dim);

  SparseMatrix<double> Q = (-L).eval();

  if(data.with_dynamics)
  {
    const double h = data.h;
    assert(h != 0);
    SparseMatrix<double> M;
    massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,data.M);
    const double dw = (1./data.ym)*(h*h);
    SparseMatrix<double> DQ = dw * 1./(h*h)*data.M;
    Q += DQ;
    // Dummy external forces
    data.f_ext = MatrixXd::Zero(n,data.dim);
    data.vel = MatrixXd::Zero(n,data.dim);
  }

  return min_quad_with_fixed_precompute(
    Q,b,SparseMatrix<double>(),true,data.solver_data);
}




bool custom_arap_solve(
    const Eigen::MatrixXd& bc,
    igl::ARAPData& data,
    Eigen::MatrixXd& U,
    Eigen::MatrixXd & rotations)
{
  using namespace Eigen;
  using namespace std;
  assert(data.b.size() == bc.rows());
  assert(U.size() != 0 && "U cannot be empty");
  assert(U.cols() == data.dim && "U.cols() match data.dim");
  if (bc.size() > 0) {
    assert(bc.cols() == data.dim && "bc.cols() match data.dim");
  }
  const int n = data.n;
  int iter = 0;
  // changes each arap iteration
  MatrixXd U_prev = U;
  // doesn't change for fixed with_dynamics timestep
  MatrixXd U0;
  if(data.with_dynamics)
  {
    U0 = U_prev;
  }
  while(iter < data.max_iter)
  {
    U_prev = U;
    // enforce boundary conditions exactly
    for(int bi = 0;bi<bc.rows();bi++)
    {
      U.row(data.b(bi)) = bc.row(bi);
    }

    const auto & Udim = U.replicate(data.dim,1);
    assert(U.cols() == data.dim);
    // As if U.col(2) was 0
    MatrixXd S = data.CSM * Udim;
    // THIS NORMALIZATION IS IMPORTANT TO GET SINGLE PRECISION SVD CODE TO WORK
    // CORRECTLY.
    S /= S.array().abs().maxCoeff();

    const int Rdim = data.dim;
    MatrixXd R(Rdim,data.CSM.rows());
    if(R.rows() == 2)
    {
      igl::fit_rotations_planar(S,R);
    }else
    {
      igl::fit_rotations(S,true,R);
//#ifdef __SSE__ // fit_rotations_SSE will convert to float if necessary
//      fit_rotations_SSE(S,R);
//#else
//      fit_rotations(S,true,R);
//#endif
    }
    //for(int k = 0;k<(data.CSM.rows()/dim);k++)
    //{
    //  R.block(0,dim*k,dim,dim) = MatrixXd::Identity(dim,dim);
    //}


    // Number of rotations: #vertices or #elements
    int num_rots = data.K.cols()/Rdim/Rdim;
    // distribute group rotations to vertices in each group
    MatrixXd eff_R;
    if(data.G.size() == 0)
    {
      // copy...
      eff_R = R;
    }else
    {
      eff_R.resize(Rdim,num_rots*Rdim);
      for(int r = 0;r<num_rots;r++)
      {
        eff_R.block(0,Rdim*r,Rdim,Rdim) =
          R.block(0,Rdim*data.G(r),Rdim,Rdim);
      }
    }
    rotations = eff_R;

    MatrixXd Dl;
    if(data.with_dynamics)
    {
      assert(data.M.rows() == n &&
        "No mass matrix. Call arap_precomputation if changing with_dynamics");
      const double h = data.h;
      assert(h != 0);
      //Dl = 1./(h*h*h)*M*(-2.*V0 + Vm1) - fext;
      // data.vel = (V0-Vm1)/h
      // h*data.vel = (V0-Vm1)
      // -h*data.vel = -V0+Vm1)
      // -V0-h*data.vel = -2V0+Vm1
      const double dw = (1./data.ym)*(h*h);
      Dl = dw * (1./(h*h)*data.M*(-U0 - h*data.vel) - data.f_ext);
    }

    VectorXd Rcol;
    igl::columnize(eff_R,num_rots,2,Rcol);
    VectorXd Bcol = -data.K * Rcol;
    assert(Bcol.size() == data.n*data.dim);
    for(int c = 0;c<data.dim;c++)
    {
      VectorXd Uc,Bc,bcc,Beq;
      Bc = Bcol.block(c*n,0,n,1);
      if(data.with_dynamics)
      {
        Bc += Dl.col(c);
      }
      if(bc.size()>0)
      {
        bcc = bc.col(c);
      }
      min_quad_with_fixed_solve(
        data.solver_data,
        Bc,bcc,Beq,
        Uc);
      U.col(c) = Uc;
    }

    iter++;
  }
  if(data.with_dynamics)
  {
    // Keep track of velocity for next time
    data.vel = (U-U0)/data.h;
  }

  return true;
}

