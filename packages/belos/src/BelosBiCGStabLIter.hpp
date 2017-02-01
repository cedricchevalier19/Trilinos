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

#ifndef BELOS_BICGSTAB_L_ITER_HPP
#define BELOS_BICGSTAB_L_ITER_HPP

/*! \file BelosBiCGStabIter.hpp
    \brief Belos concrete class for performing the pseudo-block BiCGStab iteration.
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
#include <Teuchos_SerialDenseVector.hpp>
#include <Teuchos_SerialSpdDenseSolver.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>

/*!
  \class Belos::BiCGStabIterL

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
  struct BiCGStabLIterationState {

    /*! \brief The current residual. */
    Teuchos::RCP<const MV> R0;

    /*! \brief The initial residual. */
    Teuchos::RCP<const MV> Rhat;

    /*! \brief A * M * the first decent direction vector */
    Teuchos::RCP<const MV> U0;

    ScalarType rho_0, alpha, omega;

    BiCGStabLIterationState() : R0(Teuchos::null), Rhat(Teuchos::null), U0(Teuchos::null)
    {
      rho_0 = Teuchos::ScalarTraits<ScalarType>::one();
      alpha = Teuchos::ScalarTraits<ScalarType>::one();
      omega = Teuchos::ScalarTraits<ScalarType>::one();
    }
  };

  template<class ScalarType, class MV, class OP>
  class BiCGStabLIter : virtual public Iteration<ScalarType,MV,OP> {

  public:

    //
    // Convenience typedefs
    //
    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef OperatorTraits<ScalarType,MV,OP> OPT;
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename SCT::magnitudeType MagnitudeType;

    //! @name Constructors/Destructor
    //@{

    /*! \brief %BiCGStabIter constructor with linear problem, solver utilities, and parameter list of solver options.
     *
     * This constructor takes pointers required by the linear solver, in addition
     * to a parameter list of options for the linear solver.
     */
    BiCGStabLIter( const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem,
                          const Teuchos::RCP<OutputManager<ScalarType> > &printer,
                          const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
                          Teuchos::ParameterList &params );

    //! Destructor.
    virtual ~BiCGStabLIter() {};
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
    void initializeBiCGStabL(BiCGStabLIterationState<ScalarType,MV>& newstate);

    /*! \brief Initialize the solver with the initial vectors from the linear problem
     *  or random data.
     */
    void initialize()
    {
      BiCGStabLIterationState<ScalarType,MV> empty;
      initializeBiCGStabL(empty);
    }

    /*! \brief Get the current state of the linear solver.
     *
     * The data is only valid if isInitialized() == \c true.
     *
     * \returns A BiCGStabIterationState object containing const pointers to the current
     * solver state.
     */
    BiCGStabLIterationState<ScalarType,MV> getState() const {
      BiCGStabLIterationState<ScalarType,MV> state;
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

    int l_;
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
  BiCGStabLIter<ScalarType,MV,OP>::BiCGStabLIter(const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem,
                                                               const Teuchos::RCP<OutputManager<ScalarType> > &printer,
                                                               const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
                                                               Teuchos::ParameterList &params ):
    lp_(problem),
    om_(printer),
    stest_(tester),
    numRHS_(0),
    initialized_(false),
    iter_(0),
    l_(2)
  {

    l_ = params.get("L", 2);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Initialize this iteration object
  template <class ScalarType, class MV, class OP>
  void BiCGStabLIter<ScalarType,MV,OP>::initializeBiCGStabL(BiCGStabLIterationState<ScalarType,MV>& newstate)
  {
    // Check if there is any multivector to clone from.
    Teuchos::RCP<const MV> lhsMV = lp_->getCurrLHSVec();
    Teuchos::RCP<const MV> rhsMV = lp_->getCurrRHSVec();
    TEUCHOS_TEST_FOR_EXCEPTION((lhsMV==Teuchos::null && rhsMV==Teuchos::null),std::invalid_argument,
                       "Belos::BiCGStabLIter::initialize(): Cannot initialize state storage!");

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

    // Create convenience variable for one.
    const ScalarType one = Teuchos::ScalarTraits<ScalarType>::one();

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
                         "Belos::BiCGStabIterL::initialize(): BiCGStabStateIterState does not have initial residual.");
    }

    // The solver is initialized
    initialized_ = true;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Iterate until the status test informs us we should stop.
  template <class ScalarType, class MV, class OP>
  void BiCGStabLIter<ScalarType,MV,OP>::iterate()
  {
    using Teuchos::RCP;

    //
    // Allocate/initialize data structures
    //
    if (initialized_ == false) {
      initialize();
    }

    // Allocate memory for scalars.
    std::vector<ScalarType> res(1);
    ScalarType beta;
    ScalarType sigma;
    
    // Create convenience variable for one.
    const ScalarType one = Teuchos::ScalarTraits<ScalarType>::one();

    // Residuals
    std::vector<RCP<MV>> R(l_+1);
    for (int i = 0 ; i <= l_ ; ++i) {
      R[i] = MVT::CloneCopy(*R0_); // CC: only need R[0] = R0
    }

    // U vectors
    std::vector<RCP<MV>> U(l_+1);
    for (int i = 0 ; i <= l_ ; ++i) {
      U[i] = MVT::CloneCopy(*U0_); // CC: only need U[0] = U0
    }
    
    // Get the current solution std::vector.
    Teuchos::RCP<MV> X = lp_->getCurrLHSVec();

    // For the polynomial part
    Teuchos::RCP<Teuchos::SerialSpdDenseSolver<int, ScalarType>> z_solve = Teuchos::rcp(new Teuchos::SerialSpdDenseSolver<int, ScalarType>());
    Teuchos::RCP<Teuchos::SerialSymDenseMatrix<int, ScalarType>> Z = Teuchos::rcp(new Teuchos::SerialSymDenseMatrix<int, ScalarType>(l_));
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int, ScalarType>> B = Teuchos::rcp(new Teuchos::SerialDenseMatrix<int, ScalarType>(l_, 1));
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int, ScalarType>> Y = Teuchos::rcp(new Teuchos::SerialDenseMatrix<int, ScalarType>(l_, 1));
    
    ////////////////////////////////////////////////////////////////
    // Iterate until the status test tells us to stop.
    //
    while (stest_->checkStatus(this) != Passed) {

      // Increment the iteration
      iter_++;
      // rho_0 = - omega * rho_0
      rho_0_ *= (-omega_);

      for (int j = 0 ; j < l_ ; j++) {
	//rho_1 = <R_j, Rhat_>, rho_1 = res
        MVT::MvDot(*(R[j]), *Rhat_, res);
      
        // beta = ( rho_1 / rho_0 ) (alpha)
	beta = (res[0] / rho_0_) * (alpha_);
        rho_0_ = res[0];
	
        for (int i = 0 ; i <= j ; i++) {
	  // U_i = R_i - beta * U_i
	  MVT::MvAddMv(one, *(R[i]), -beta, *(U[i]), *(U[i]));
        }

        // U_{j+1} = K\(A U_j)
	lp_->apply(*(U[j]), *(U[j+1]));

	// sigma = <U_{j+1}, Rhat_>
        MVT::MvDot(*(U[j+1]), *Rhat_, res);
	// alpha = rho_1 / sigma
	// CC: here rho_0 == rho_1
	alpha_ = rho_0_ / res[0];
	
	// x = x + alpha*u_0
	MVT::MvAddMv(one, *X, alpha_, *(U[0]), *X);

	for (int  i = 0 ; i <= j ; ++i ) {
	  // R_i = R_i - alpha*U_{i+1}
	  MVT::MvAddMv(one, *(R[i]), -alpha_, *(U[i+1]), *(R[i]));
	}

	// r_{j+1} = K\Ar_j
	lp_->apply(*(R[j]), *(R[j+1]));
      }

      //------------------
      // Polynomial Part

      for (int i = 0 ; i < l_ ; ++i) {
	for (int j = 0 ; j <= i ; ++j ) {
	  // Z[i,j] = <r_j, r_i> , (i,j) \in [1,l]
	  MVT::MvDot(*(R[j+1]), *(R[i+1]), res);
	  (*Z)(i,j) = res[0] ;
	}
	// Y[i] = <r_0, r_i>
	MVT::MvDot(*(R[0]), *(R[i+1]), res);
	(*B)(i,0) = res[0];
      }

      // y = Z\y
      z_solve->setMatrix(Z);
      z_solve->setVectors(Y, B);
      z_solve->solve();

      // omega = Y[l]
      omega_ = (*Y)(l_-1, 0);

      // Update
      for (int i = 0 ; i < l_ ; ++i ){
	ScalarType scale = (*Y)(i,0);
	// u_0 = u_0 - y[i]u_i
	MVT::MvAddMv(one, *(U[0]), -scale, *(U[i+1]), *(U[0]));
	// x = x + y[i] r_{i-1}
	MVT::MvAddMv(one, *X, scale, *(R[i]), *X);
	// r_0 = r_0 - y[i] r_i
       	MVT::MvAddMv(one, *(R[0]), -scale, *(R[i+1]), *(R[0]));
      }

      MVT::Assign(*(R[0]), *R0_);
      MVT::Assign(*(U[0]), *U0_);
    } // end while (sTest_->checkStatus(this) != Passed)
  }

} // end Belos namespace

#endif /* BELOS_BICGSTABL_ITER_HPP */
