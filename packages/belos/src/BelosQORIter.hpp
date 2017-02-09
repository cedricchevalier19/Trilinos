//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER

#ifndef BELOS_QOR_ITER_HPP
#define BELOS_QOR_ITER_HPP

/*! \file BelosQORIter.hpp
    \brief Belos concrete class for performing the Q-OR iteration.
*/

#include <algorithm>

#include "BelosConfigDefs.hpp"
#include "BelosTypes.hpp"


#include "BelosLinearProblem.hpp"
#include "BelosMatOrthoManager.hpp"
#include "BelosOutputManager.hpp"
#include "BelosStatusTest.hpp"
#include "BelosOperatorTraits.hpp"
#include "BelosMultiVecTraits.hpp"

#include <Teuchos_BLAS.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialSpdDenseSolver.hpp>
#include <Teuchos_SerialDenseVector.hpp>
#include <Teuchos_SerialDenseHelpers.hpp>
#include <Teuchos_SerialSpdDenseSolver.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>

/*!
  \class Belos::BelosQORIter

  \brief This class implements the scalar BiCGStab(l) iteration.
  Multiple RHS are not handled.


  \ingroup belos_solver_framework

  \author Cedric Chevalier <cedric.chevalier@cea.fr>
*/

namespace Belos {

  //! @name BiCGStabIterationL Structures
  //@{

  /** \brief Structure to contain pointers to BiCGStabIteration state variables.
   *
   * This struct is utilized by BiCGStabIterationL::initialize() and BiCGStabIterationL::getState().
   */
  template <class ScalarType, class MV>
  struct QORIterationState {

    /*! \brief The current residual. */
    Teuchos::RCP<const MV> R0;

    /*! \brief The initial residual. */
    Teuchos::RCP<const MV> Rhat;

    /*! \brief A * M * the first decent direction vector */
    Teuchos::RCP<const MV> U0;

    ScalarType rho_0, alpha, omega;

    QORIterationState() : R0(Teuchos::null), Rhat(Teuchos::null), U0(Teuchos::null)
    {
      rho_0 = Teuchos::ScalarTraits<ScalarType>::one();
      alpha = Teuchos::ScalarTraits<ScalarType>::one();
      omega = Teuchos::ScalarTraits<ScalarType>::one();
    }
  };



  template<class ScalarType, class MV, class OP>
  class QORIter : virtual public Iteration<ScalarType,MV,OP> {

  public:

    //
    // Convenience typedefs
    //
    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef OperatorTraits<ScalarType,MV,OP> OPT;
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename SCT::magnitudeType MagnitudeType;

    typedef Teuchos::SerialDenseVector<int, ScalarType> LVector;
    typedef Teuchos::SerialDenseMatrix<int, ScalarType> LMatrix;

    //! @name Constructors/Destructor
    //@{

    /*! \brief %BiCGStabIter constructor with linear problem, solver utilities, and parameter list of solver options.
     *
     * This constructor takes pointers required by the linear solver, in addition
     * to a parameter list of options for the linear solver.
     */
    QORIter( const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem,
			  const Teuchos::RCP<OutputManager<ScalarType> > &printer,
			  const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
			  Teuchos::ParameterList &params );

    //! Destructor.
    virtual ~QORIter() {};
    //@}


    //! @name Solver methods
    //@{

    /*! \brief This method performs BiCGStab iterations on each linear system until the status
     * test indicates the need to stop or an error occurs (in which case, an
     * std::exception is thrown).
     *
     * iterate() will first determine whether the solver is initialized; if
     * not, it will call initialize() using default arguments. After
     * initialization, the solver performs BiCGStab iterations until the
     * status test evaluates as ::Passed, at which point the method returns to
     * the caller.
     *
     * The status test is queried at the beginning of the iteration.
     *
     */
    void iterate();

    /*! \brief Initialize the solver to an iterate, providing a complete state.
     *
     * The %BiCGStabIter contains a certain amount of state, consisting of the current
     * direction vectors and residuals.
     *
     * initialize() gives the user the opportunity to manually set these,
     * although this must be done with caution, abiding by the rules given
     * below.
     *
     * \post
     * <li>isInitialized() == \c true (see post-conditions of isInitialize())
     *
     * The user has the option of specifying any component of the state using
     * initialize(). However, these arguments are assumed to match the
     * post-conditions specified under isInitialized(). Any necessary component of the
     * state not given to initialize() will be generated.
     *
     * \note For any pointer in \c newstate which directly points to the multivectors in
     * the solver, the data is not copied.
     */
    void initializeQOR(QORIterationState<ScalarType,MV>& newstate);

    /*! \brief Initialize the solver with the initial vectors from the linear problem
     *  or random data.
     */
    void initialize()
    {
      QORIterationState<ScalarType,MV> empty;
      initializeQOR(empty);
    }

    /*! \brief Get the current state of the linear solver.
     *
     * The data is only valid if isInitialized() == \c true.
     *
     * \returns A BiCGStabIterationState object containing const pointers to the current
     * solver state.
     */
    QORIterationState<ScalarType,MV> getState() const {
      QORIterationState<ScalarType,MV> state;
      state.R0 = R0_;
      state.Rhat = Rhat_;
      state.U0 = U0_;
      state.rho_0 = rho_0_;
      state.alpha = alpha_;
      state.omega = omega_;
      return state;
    }

    //@}


    //! @name Status methods
    //@{

    //! \brief Get the current iteration count.
    int getNumIters() const { return iter_; }

    //! \brief Reset the iteration count.
    void resetNumIters( int iter = 0 ) { iter_ = iter; }

    //! Get the norms of the residuals native to the solver.
    //! \return A std::vector of length blockSize containing the native residuals.
    // amk TODO: are the residuals actually being set?  What is a native residual?
    Teuchos::RCP<const MV> getNativeResiduals( std::vector<MagnitudeType> *norms ) const { return R0_; }

    //! Get the current update to the linear system.
    /*! \note This method returns a null pointer because the linear problem is current.
    */
    // amk TODO: what is this supposed to be doing?
    Teuchos::RCP<MV> getCurrentUpdate() const { return Teuchos::null; }

    //@}

    //! @name Accessor methods
    //@{

    //! Get a constant reference to the linear problem.
    const LinearProblem<ScalarType,MV,OP>& getProblem() const { return *lp_; }

    //! Get the blocksize to be used by the iterative solver in solving this linear problem.
    int getBlockSize() const { return 1; }

    //! \brief Set the blocksize.
    void setBlockSize(int blockSize) {
      TEUCHOS_TEST_FOR_EXCEPTION(blockSize!=1,std::invalid_argument,
			 "Belos::BiCGStabIter::setBlockSize(): Cannot use a block size that is not one.");
    }

    //! States whether the solver has been initialized or not.
    bool isInitialized() { return initialized_; }

    //@}

  private:

    //
    // Classes inputed through constructor that define the linear problem to be solved.
    //
    const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> >    lp_;
    const Teuchos::RCP<OutputManager<ScalarType> >          om_;
    const Teuchos::RCP<StatusTest<ScalarType,MV,OP> >       stest_;

    //
    // Algorithmic parameters
    //
    // numRHS_ is the current number of linear systems being solved.
    int numRHS_;

    //
    // Current solver state
    //
    // initialized_ specifies that the basis vectors have been initialized and the iterate() routine
    // is capable of running; _initialize is controlled  by the initialize() member method
    // For the implications of the state of initialized_, please see documentation for initialize()
    bool initialized_;

    // Current number of iterations performed.
    int iter_;

    int m_;

    //
    // State Storage
    //
    // Initial residual
    Teuchos::RCP<MV> Rhat_;
    //
    // Residual
    Teuchos::RCP<MV> R0_;
    //
    // Operator applied to preconditioned direction vector 1
    Teuchos::RCP<MV> U0_;
    //
    ScalarType rho_0_, alpha_, omega_;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Constructor.
  template<class ScalarType, class MV, class OP>
  QORIter<ScalarType,MV,OP>::QORIter(const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem,
							       const Teuchos::RCP<OutputManager<ScalarType> > &printer,
							       const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
							       Teuchos::ParameterList &params ):
    lp_(problem),
    om_(printer),
    stest_(tester),
    numRHS_(0),
    initialized_(false),
    iter_(0),
    m_(10)
  {
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Initialize this iteration object
  template <class ScalarType, class MV, class OP>
  void QORIter<ScalarType,MV,OP>::initializeQOR(QORIterationState<ScalarType,MV>& newstate)
  {
    // Check if there is any multivector to clone from.
    Teuchos::RCP<const MV> lhsMV = lp_->getCurrLHSVec();
    Teuchos::RCP<const MV> rhsMV = lp_->getCurrRHSVec();
    TEUCHOS_TEST_FOR_EXCEPTION((lhsMV==Teuchos::null && rhsMV==Teuchos::null),std::invalid_argument,
		       "Belos::QORIter::initialize(): Cannot initialize state storage!");

    // Get the multivector that is not null.
    Teuchos::RCP<const MV> tmp = ( (rhsMV!=Teuchos::null)? rhsMV: lhsMV );

    // Get the number of right-hand sides we're solving for now.
    int numRHS = MVT::GetNumberVecs(*tmp);
    numRHS_ = numRHS;

    if (numRHS_ != 1)
      throw "Too many RHS";

    // Initialize the state storage
    // If the subspace has not be initialized before or has changed sizes, generate it using the LHS or RHS from lp_.
    if (Teuchos::is_null(R0_) || MVT::GetNumberVecs(*R0_)!=numRHS_) {
      R0_ = MVT::Clone( *tmp, numRHS_ );
      Rhat_ = MVT::Clone( *tmp, numRHS_ );
      U0_ = MVT::Clone( *tmp, numRHS_ );
    }

    // NOTE:  In BiCGStabIter R_, the initial residual, is required!!!
    //
    std::string errstr("Belos::BlockPseudoCGIter::initialize(): Specified multivectors must have a consistent length and width.");

    if (!Teuchos::is_null(newstate.R0)) {

      TEUCHOS_TEST_FOR_EXCEPTION( MVT::GetGlobalLength(*newstate.R0) != MVT::GetGlobalLength(*R0_),
			  std::invalid_argument, errstr );
      TEUCHOS_TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.R0) != numRHS_,
			  std::invalid_argument, errstr );

      // Copy residual vectors from newstate into R
      if (newstate.R0 != R0_) {
	// Assigned by the new state
	MVT::Assign(*newstate.R0, *R0_);
      }
      else {
	// Computed
	lp_->computeCurrResVec(R0_.get());
      }

      // Set Rhat
      if (!Teuchos::is_null(newstate.Rhat) && newstate.Rhat != Rhat_) {
	// Assigned by the new state
	MVT::Assign(*newstate.Rhat, *Rhat_);
      }
      else {
	// Set to be the initial residual
	MVT::Assign(*lp_->getInitResVec(), *Rhat_);
      }

      // Set U0
      if (!Teuchos::is_null(newstate.U0) && newstate.U0 != U0_) {
	// Assigned by the new state
	MVT::Assign(*newstate.U0, *U0_);
      }
      else {
	// Initial V = 0
	MVT::MvInit(*U0_);
      }


      // Set rho_old
      rho_0_ = newstate.rho_0;


      // Set alpha
      alpha_ = newstate.alpha;

      // Set omega
      omega_ = newstate.omega;

    }
    else {

      TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(newstate.R0),std::invalid_argument,
			 "Belos::BelosQORIter::initialize(): BiCGStabStateIterState does not have initial residual.");
    }

    // The solver is initialized
    initialized_ = true;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Iterate until the status test informs us we should stop.
  template <class ScalarType, class MV, class OP>
  void QORIter<ScalarType,MV,OP>::iterate()
  {
    using Teuchos::RCP;
    using Teuchos::rcp;

    //
    // Allocate/initialize data structures
    //
    if (initialized_ == false) {
      initialize();
    }


    RCP<MV> V;
    RCP<MV> V_A;
    RCP<MV> V_tilde;
    RCP<MV> PV;

    RCP<LMatrix> L = rcp(new LMatrix(m_, m_));
    RCP<LMatrix> H = rcp(new LMatrix(m_, m_));

    ScalarType omega;
    ScalarType alpha;

    RCP<LVector> nu = rcp(new LVector(m_));
    RCP<LVector> vta = rcp(new LVector(m_));
    RCP<LVector> vv = rcp(new LVector(m_));
    RCP<LVector> y = rcp(new LVector(m_));

    // Allocate memory for scalars.
    std::vector<ScalarType> res(1);

    // Create convenience variable for one.
    const ScalarType one  = Teuchos::ScalarTraits<ScalarType>::one();
    const ScalarType zero = Teuchos::ScalarTraits<ScalarType>::zero();

    int k;

    ////////////////////////////////////////////////////////////////
    // Iterate until the status test tells us to stop.
    //
    while (stest_->checkStatus(this) != Passed) {
      iter_++;

      // vv_k = tV_k-1 v_k
      Teuchos::Range1D range_k1 (0, k-1);
      std::vector<int> column(1);
      column[0] = k;
      RCP<MV> mV_k1 = MVT::CloneView(V, range_k1);
      RCP<MV> V_k = MVT::CloneView(V, column);
      LVector vv_k(Teuchos::View, k-1);
      MVT::MvTransMv(one, *mV_k1, *V_k, vv_k);

      // vtA_k = tV_k vA_k
      LVector vta_k(Teuchos::View, *vta);
      Teuchos::Range1D range_k (0, k);
      RCP<MV> mV_k = MVT::CloneView(*V, range_k);
      MVT::MvTransMv(one, *mV_k, *V_A, vta_k);

      // Extract submatrix L_k-1
      LMatrix L_k1(Teuchos::View, *L, k-1, k-1);
      LVector l_k = Teuchos::getCol<int, ScalarType>(Teuchos::View, *L, k);

      // l_k = L_k-1 vv_k
      l_k.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, one, L_k1, vv_k, zero);

      // ty_k = tl_k L_k-1
      // ie: y_k = tL_k-1 l_k
      LVector y_k(Teuchos::View, *y, k);
      y_k.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, one, L_k1, *l_k, zero);

      // tpv_k = ty_k tVk-1
      // ie: pv_k = Vk-1 y_k
      RCP<MV> PV_k = MVT::CloneView(*PV, range_k);
      MVT::MvTimesMatAddMv(one, *mV_k1, y_k, zero, *PV_k);

      // l_kk = norm(vk-pv_k)
      MVT::MvAddMv(one, *mV_k, -one, *PV_k, *PV_k);
      MVT::MvNorm(*PV_k, res);
      ScalarType lkk = res[0];

      // L_k = L_k-1 union { -1/l_kk*ty_k , 1/l_kk}
      LMatrix L_k(Teuchos::View, *L, k, k);
      for (int i = 0 ; i < k-1 ; ++i) {
	L_k(k-1, i) = - y_k(i)/lkk;
      }
      L_k(k-1, k-1) = 1/lkk;

      // l_nu = L_k nu << recursive trick !
      LVector nu_k1(Teuchos::View, *nu, k-1);

      LVector l_nu(k);
      // l_nu = ( L_k-1*nu_k-1 )
      //        ( 1/l_kk (-tl_k L_k-1 nu_k-1 + vu_1,k))
      l_nu.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, one, L_k1, nu_k1, zero);
      l_nu(k) = (-l_k.dot(nu_k1) + l_nu(k))/lkk;

      // l_A = L_k vtA_k
      LVector l_A(k);
      l_A.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, one, L_k, vta_k, zero);

      // omega = dot(l_A, l_nu)
      omega = l_A.dot(l_nu);

      // alpha = |vA_k|^2 - |l_A|^2
      MVT::MvDot(*V_A, *V_A, res);
      alpha = res[0] - l_A.dot(l_A);

      // h(1:k, k) = tL_k (l_A + alpha/omega l_nu)
      LMatrix H_k(Teuchos::View, *H, k);
      LVector h_k = Teuchos::getCol<int, ScalarType>(Teuchos::View, H_k, k);

      h_k = l_A + alpha/omega * l_nu;
      h_k.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, one, L_k, h_k, zero);

      // v_tilde = vA_k - V_k h(1:k,k)
      V_tilde = MVT::CloneView(V_A);
      MVT::MvTimesMatAddMv(-one, V_k, h_k, one, *V_tilde);
      // h(k_+1, k) = |v_tilde|^2
      MVT::MvNorm(*V_tilde, res, TwoNorm);
      (*H)(k+1, k) = res[0];

      // nu(1, k+1) = -1/(h(k+1,k) tnu h(1:k, k)
      LVector nu_k(Teuchos::View, *nu, k);
      (*nu)(1, k+1) = -1/(*H)(k+1, k) * nu_k.dot(h_k);

      // nu = (nu(1,1) ... nu(1,k+1))
      // Nothing to do.

      // v_k+1 = 1/h(k+1, k) v_tilde
      RCP<MV> V_kp1 = MVT::CloneView(V, k+1);
      MVT::MvAddMv(1/(*H)(k+1, k), *V_tilde, zero, *V_tilde, V_kp1);

      // vA_k+1 = A v_k+1
      // CC: Check if it can work with a preconditionner
      lp_->apply(V_A, V_kp1);

    } // end while (sTest_->checkStatus(this) != Passed)
  }


} // end Belos namespace

#endif /* BELOS_BICGSTABL_ITER_HPP */
