#ifndef COMPILATION_H_
#define COMPILATION_H_

#include "core/common.hpp"
#include "core/node_manager.hpp"

namespace zhetapi {

// Forward declarations
class Engine;

namespace lang {

node_manager compile_block(Engine *, const std::string &, Args, Pardon &);

}

}

#endif
