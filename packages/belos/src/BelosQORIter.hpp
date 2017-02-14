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

    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename SCT::magnitudeType MagnitudeType;
    typedef Teuchos::SerialDenseMatrix<int, ScalarType> LMatrix;
    typedef Teuchos::SerialDenseVector<int, ScalarType> LVector;

    /*! \brief The current dimension of the reduction.
     *
     * This should always be equal to PseudoBlockGmresIter::getCurSubspaceDim()
     */
    int curDim;

    /*! \brief The current Krylov basis. Multivector of size numBlocks */
    Teuchos::RCP<MV> V;
    /*! \brief The current Av_k vector. */
    Teuchos::RCP<MV> VA;

    /*! \brief The current Hessenberg matrix.
     *
     * The \c curDim by \c curDim leading submatrix of H is the
     * projection of problem->getOperator() by the first \c curDim vectors in V.
     */
    Teuchos::RCP<LMatrix> H;

    /*! \brief Auxiliary Lower triangular matrix used to fill up the Hessenberg matrix */
    Teuchos::RCP<LMatrix> L;

    Teuchos::RCP<LVector> nu;

    ScalarType omega;
    ScalarType alpha;

    QORIterationState() : curDim(0), V(Teuchos::null), VA(Teuchos::null), H(Teuchos::null)
			, L(Teuchos::null), nu(Teuchos::null)
    {}
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
    typedef int OrdinalType; // No ordinal information from Traits !

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
    void initialize(QORIterationState<ScalarType,MV>& newstate);

    /*! \brief Initialize the solver with the initial vectors from the linear problem
     *  or random data.
     */
    void initialize()
    {
      QORIterationState<ScalarType,MV> empty;
      initialize(empty);
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
    Teuchos::RCP<const MV> getNativeResiduals( std::vector<MagnitudeType> *norms ) const;

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


    //! Get the dimension of the search subspace used to generate the current solution to the linear problem.
    int getCurSubspaceDim() const {
      if (!initialized_) return 0;
      return curDim_ + 1; // +1 because 0..curDim_
    };

    //! Get the maximum dimension allocated for the search subspace.
    int getMaxSubspaceDim() const { return numBlocks_; }

    //! \brief Set the maximum number of blocks used by the iterative solver.
    void setNumBlocks(int numBlocks) { numBlocks_ = numBlocks; }

    //! Get the maximum number of blocks used by the iterative solver in solving this linear problem.
    int getNumBlocks() const { return numBlocks_; }


    //! \brief Set the blocksize.
    void setBlockSize(int blockSize) {
      TEUCHOS_TEST_FOR_EXCEPTION(blockSize!=1,std::invalid_argument,
			 "Belos::BiCGStabIter::setBlockSize(): Cannot use a block size that is not one.");
    }

    //! States whether the solver has been initialized or not.
    bool isInitialized() { return initialized_; }

    //@}

  private:
    //! Restore a state and return whether a formal initialization is necessary.
    bool _simplyRestore(QORIterationState<ScalarType,MV>& newstate);

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

    int numBlocks_;

    int curDim_;

    //
    // State Storage
    //
    Teuchos::RCP<MV> V_;
    Teuchos::RCP<MV> VA_;

    Teuchos::RCP<LMatrix> L_;
    Teuchos::RCP<LMatrix> H_;

    ScalarType omega_;
    ScalarType alpha_;

    Teuchos::RCP<LVector> nu_;
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
    numBlocks_(10)
  {
  }


  template <class ScalarType, class MV, class OP>
  bool QORIter<ScalarType,MV,OP>::_simplyRestore(QORIterationState<ScalarType,MV>& newstate)
  {
    bool simple_restore = true;
    if (newstate.V.is_null()) {
      V_ = MVT::Clone(*lp_->getRHS(), numBlocks_); // V is not initialized

      // Copy RHS in the first column.
      std::vector<int> column(1);
      column[0] = 0;
      Teuchos::RCP<MV> V0 = MVT::CloneViewNonConst(*V_, column);
      MVT::MvAddMv(SCT::one(), *(lp_->getRHS()), SCT::zero(), *V0, *V0);

      simple_restore = false;
    }
    else {
      V_ = newstate.V;
    }

    if (newstate.VA.is_null()) {
      VA_ = MVT::Clone(*V_, 1);
      simple_restore = false;
    }
    else {
      VA_ = newstate.VA;
    }

    if (newstate.H.is_null()) {
      H_ = Teuchos::rcp<LMatrix>(new LMatrix(numBlocks_, numBlocks_));
      simple_restore = false;
    }
    else {
      H_ = newstate.H;
    }

    if (newstate.L.is_null()) {
      L_ = Teuchos::rcp<LMatrix>(new LMatrix(numBlocks_, numBlocks_));
      simple_restore = false;
    }
    else {
      L_ = newstate.L;
    }

    if (newstate.nu.is_null()) {
      nu_ = Teuchos::rcp<LVector>(new LVector(numBlocks_));
      simple_restore = false;
    }
    else {
      nu_ = newstate.nu;
    }

    if (omega_ != newstate.omega) simple_restore = false;
    if (alpha_ != newstate.alpha) simple_restore = false;

    omega_ = newstate.omega;
    alpha_ = newstate.alpha;
    if (newstate.curDim != curDim_) {
      simple_restore =false;
      curDim_ = newstate.curDim;
    }

    return simple_restore;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Initialize this iteration object
  template <class ScalarType, class MV, class OP>
  void QORIter<ScalarType,MV,OP>::initialize(QORIterationState<ScalarType,MV>& newstate)
  {
    const ScalarType one = Teuchos::ScalarTraits<ScalarType>::one();
    const ScalarType zero = Teuchos::ScalarTraits<ScalarType>::zero();

    if (_simplyRestore(newstate) // We have not done any new allocation
	&& ((newstate.curDim != 0) && (newstate.curDim != numBlocks_ - 1))) { // we still can compute blocks
      // We do not recompute all the initialization.
      initialized_ = true;
      return;
    }

    curDim_ = 1;

    std::vector<int> column(1);
    column[0] = 0;

    Teuchos::RCP<MV> V0 = MVT::CloneViewNonConst(*V_, column);

    MVT::MvPrint(*V0, std::cerr);
    std::vector<MagnitudeType> norm(1);
    MVT::MvNorm(*V0, norm, TwoNorm);
    MVT::MvScale(*V0, 1/norm[0]);

    lp_->apply(*V0, *VA_);

    L_ = rcp(new LMatrix(numBlocks_, numBlocks_, true)); // put zero everywhere
    (*L_)(0,0) = one;

    std::vector<ScalarType> dot(1);
    MVT::MvDot(*V0, *VA_, dot);
    omega_ = dot[0];

    MVT::MvDot(*VA_, *VA_, dot);
    alpha_ = dot[0] - omega_*omega_;

    H_ = rcp(new LMatrix(numBlocks_, numBlocks_, true)); // put zero everywhere
    (*H_)(0,0) = omega_ + alpha_/omega_;
    (*nu_)(0) = one; // CC: not in the paper

    Teuchos::RCP<MV> V_tilde = MVT::Clone(*VA_, 1);
    MVT::MvAddMv(one, *VA_, -(*H_)(0,0), *V0, *V_tilde);

    MVT::MvNorm(*V_tilde, norm, TwoNorm);
    (*H_)(1,0) = norm[0];

    column[0] = 1;
    Teuchos::RCP<MV> V1 = MVT::CloneViewNonConst(*V_, column);
    MVT::MvAddMv(zero, *V1, (*H_)(1,0), *V_tilde, *V1);

    lp_->apply(*V1, *VA_);

    (*nu_)(1) = - (*H_)(0,0)/(*H_)(1,0);

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

    RCP<MV> PV_k = MVT::Clone(*V_, 1);
    RCP<MV> V_tilde;

    RCP<LVector> vta = rcp(new LVector(numBlocks_));
    RCP<LVector> vv = rcp(new LVector(numBlocks_));
    RCP<LVector> y = rcp(new LVector(numBlocks_));
    RCP<LVector> lk = rcp(new LVector(numBlocks_));

    // Allocate memory for scalars.
    std::vector<ScalarType> res(1);

    // Create convenience variable for one.
    const ScalarType one  = Teuchos::ScalarTraits<ScalarType>::one();
    const ScalarType zero = Teuchos::ScalarTraits<ScalarType>::zero();

    ////////////////////////////////////////////////////////////////
    // Iterate until the status test tells us to stop.
    //
    while ((stest_->checkStatus(this) != Passed) && (curDim_ < (numBlocks_ - 1))) {
      iter_++;

      // vv_k = tV_k-1 v_k
      Teuchos::Range1D range_k1 (0, curDim_-1);
      std::vector<int> column(1);
      column[0] = curDim_;
      RCP<const MV> mV_k1 = MVT::CloneView(*V_, range_k1);
      RCP<const MV> V_k = MVT::CloneView(*V_, column);
      LVector vv_k(Teuchos::View, *vv, curDim_); // dim = k (0..k-1)
      MVT::MvTransMv(one, *mV_k1, *V_k, vv_k);

      // vtA_k = tV_k vA_k
      LVector vta_k(Teuchos::View, *vta, curDim_+1);  // dim = k+1 (0..k)
      Teuchos::Range1D range_k (0, curDim_);
      RCP<const MV> mV_k = MVT::CloneView(*V_, range_k);
      MVT::MvTransMv(one, *mV_k, *VA_, vta_k);

      // Extract submatrix L_k-1
      LMatrix L_k1(Teuchos::View, *L_, curDim_, curDim_);  // dim = k
      LVector l_k(Teuchos::View, *lk, curDim_); // Take only the k-1 first elements.

      // l_k = L_k-1 vv_k
      l_k.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, one, L_k1, vv_k, zero);

      // ty_k = tl_k L_k-1
      // ie: y_k = tL_k-1 l_k
      LVector y_k(Teuchos::View, *y, curDim_); // dim = k
      y_k.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, one, L_k1, l_k, zero);

      // tpv_k = ty_k tVk-1
      // ie: pv_k = Vk-1 y_k
      MVT::MvTimesMatAddMv(one, *mV_k1, y_k, zero, *PV_k);

      // l_kk = norm(vk-pv_k)
      MVT::MvAddMv(one, *V_k, -one, *PV_k, *PV_k);
      MVT::MvNorm(*PV_k, res);
      ScalarType lkk = res[0];

      if (lkk <= SCT::sfmin()) {
	throw "lkk is null";
      }

      // L_k = L_k-1 union { -1/l_kk*ty_k , 1/l_kk}
      LMatrix L_k(Teuchos::View, *L_, curDim_+1, curDim_+1); // dim = k+1
      for (int i = 0 ; i < curDim_ - 1 ; ++i) {
	L_k(curDim_, i) = - y_k(i)/lkk;
      }
      L_k(curDim_, curDim_) = 1/lkk;

      // l_nu = L_k nu << recursive trick !
      LVector nu_k1(Teuchos::View, *nu_, curDim_); // dim = k

      LVector l_nu(curDim_+1); // dim = k
      LVector l_nu1(Teuchos::View, l_nu, curDim_); // dim = k
      // l_nu = ( L_k-1*nu_k-1 )
      //        ( 1/l_kk (-tl_k L_k-1 nu_k-1 + vu_1,k))
      l_nu1.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, one, L_k1, nu_k1, zero);
      l_nu(curDim_) = (-l_k.dot(l_nu1) + (*nu_)(curDim_))/lkk;

      // l_A = L_k vtA_k
      LVector l_A(curDim_+1);
      l_A.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, one, L_k, vta_k, zero);

      // omega = dot(l_A, l_nu)
      omega_ = l_A.dot(l_nu);

      if (std::abs(omega_) <= SCT::sfmin()) {
	throw "omega is null";
      }

      // alpha = |vA_k|^2 - |l_A|^2
      MVT::MvDot(*VA_, *VA_, res);
      alpha_ = res[0] - l_A.dot(l_A);

      // h(1:k, k) = tL_k (l_A + alpha/omega l_nu)
      LMatrix H_k(Teuchos::View, *H_, curDim_+1, curDim_+1); // dim = k+1
      LVector h_k = Teuchos::getCol<int, ScalarType>(Teuchos::View, H_k, curDim_); // dim = k+1, select column k

      h_k = l_A;
      l_nu *= alpha_/omega_;
      h_k += l_nu;
      h_k.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, one, L_k, h_k, zero);

      // v_tilde = vA_k - V_k h(1:k,k)
      // column[0] == k
      column[0] = 0; // Reuse VA_
      V_tilde = MVT::CloneViewNonConst(*VA_, column);
      MVT::MvTimesMatAddMv(-one, *mV_k, h_k, one, *V_tilde);
      // h(k_+1, k) = |v_tilde|^2
      MVT::MvNorm(*V_tilde, res, TwoNorm);
      (*H_)(curDim_+1, curDim_) = res[0];

      // nu(1, k+1) = -1/(h(k+1,k) tnu h(1:k, k)
      LVector nu_k(Teuchos::View, *nu_, curDim_+1); // dim = k+1 : 0..k
      (*nu_)(curDim_+1) = -1/(*H_)(curDim_+1, curDim_) * nu_k.dot(h_k);

      // nu = (nu(1,1) ... nu(1,k+1))
      // Nothing to do.

      // v_k+1 = 1/h(k+1, k) v_tilde
      column[0] = curDim_ + 1;
      RCP<MV> V_kp1 = MVT::CloneViewNonConst(*V_, column);
      MVT::MvAddMv(1/(*H_)(curDim_+1, curDim_), *V_tilde, zero, *V_tilde, *V_kp1);

      // vA_k+1 = A v_k+1
      // CC: Check if it can work with a preconditionner
      lp_->apply(*VA_, *V_kp1);

      curDim_++;
    } // end while (sTest_->checkStatus(this) != Passed)
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Get the native residuals stored in this iteration.
  // Note:  No residual std::vector will be returned by Gmres.
  template <class ScalarType, class MV, class OP>
  Teuchos::RCP<const MV> QORIter<ScalarType,MV,OP>::getNativeResiduals( std::vector<MagnitudeType> *norms ) const
  {
    //
    // NOTE: Make sure the incoming std::vector is the correct size!
    //
    if (!norms) return Teuchos::null;

    if ((int)norms->size() < 1 )
      norms->resize( 1 );

    (*norms)[0] = 1/std::abs((*nu_)[curDim_]);

    return Teuchos::null;
  }



} // end Belos namespace

#endif /* BELOS_BICGSTABL_ITER_HPP */
