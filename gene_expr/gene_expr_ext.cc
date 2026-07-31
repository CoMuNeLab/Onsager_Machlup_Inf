#include "gene_expr_ext.hh"
using DState = graph_tool::GeneExprState;
#include <inference/uncertain/dynamics_base_imp.hh>

#include <boost/python.hpp>
#define GRAPH_VIEWS decltype(never_filtered_never_reversed)
#include <inference/blockmodel/graph_blockmodel.hh>
#include <inference/uncertain/dynamics.hh>
#include <inference/uncertain/dynamics_base.hh>
#include <support/graph_state.hh>

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
struct Dyn : Dynamics<BaseState> {};

template <class BaseState>
GEN_DISPATCH(dynamics_state, Dyn<BaseState>::template DynamicsState,
             DYNAMICS_STATE_params)

python::object make_gene_expr_state(python::object odynamics_state,
                                    python::object ot, python::object os,
                                    python::object oa, python::dict params)
{
    python::object state;
    block_state::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;
             dynamics_state<block_state_t>::dispatch
                 (odynamics_state,
                  [&](auto& s)
                  {
                      state = python::object(std::make_shared<DState>(params, s,
                                                                      ot, os, oa));
                  }, false);
         });
    return state;
}

BOOST_PYTHON_MODULE(libgene_expr_ext)
{
    using namespace boost::python;
    def("make_gene_expr_state", &make_gene_expr_state);
    class_<DState, bases<DStateBase>, std::shared_ptr<DState>,
           boost::noncopyable>("GeneExprState", no_init);
}
