#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <utility>

#include "api.h"
#include "random.h"
#include "tensor.h"

namespace {
using namespace ::peachygrad;
namespace py = ::pybind11;
}  // namespace

PYBIND11_MODULE(peachygrad_cc, m) {
  m.doc() = "Peachygrad!";

  // Objects.
  py::enum_<DType>(m, "dtype")
      .value("f32", DType::kF32)
      .value("i32", DType::kI32)
      .def("__repr__", [](DType dtype) {
        std::stringstream ss;
        ss << dtype;
        return ss.str();
      });

  py::class_<Shape>(m, "shape")
      .def(py::init([](py::tuple py_shape) {
        std::unique_ptr<Shape> shape = std::make_unique<Shape>();
        shape->num_dims = py_shape.size();
        for (int i = 0; i < py_shape.size(); ++i) {
          (*shape)[i] = py_shape[i].cast<int>();
        }

        return shape;
      }))
      .def("transpose", [](const Shape& shape) { return shape.transpose(); })
      .def("__len__", [](const Shape& shape) { return shape.num_dims; })
      .def("__eq__", &Shape::operator==)
      .def("__getitem__", [](const Shape& shape, size_t i) { return shape[i]; })
      .def(
          "__iter__",
          [](Shape& shape) {
            return py::make_iterator(shape.begin(), shape.end());
          },
          py::keep_alive<0, 1>())
      .def("__repr__", [](const Shape& shape) {
        std::stringstream ss;
        ss << shape;
        return ss.str();
      });

  py::class_<Tensor>(m, "Tensor")
      .def("dtype", &Tensor::dtype)
      .def("shape", &Tensor::shape)
      .def("copy", &Tensor::copy)
      .def("__eq__", &Tensor::operator==)
      .def("__repr__", [](const Tensor& tensor) {
        std::stringstream ss;
        ss << tensor;
        return ss.str();
      });

  // API.
  m.attr("f32") = DType::kF32;
  m.attr("i32") = DType::kI32;

  // Tensor
  m.def(
      "tensor", [](py::list py_list) { return tensor(py_list, DType::kF32); },
      "Initialize tensor (default f32).");
  m.def(
      "tensor",
      [](py::list py_list, DType dtype) { return tensor(py_list, dtype); },
      "Initialize tensor.");
  m.def(
      "tensor", [](py::array_t<float> np_arr) { return tensor(np_arr); },
      "Initialize tensor.");

  // Numpy
  m.def(
      "numpy", [](Tensor& t) { return numpy(t); },
      "Convert tensor to numpy.");

  // Zeros
  m.def("zeros", [](Shape& shape, DType dtype) { return zeros(shape, dtype); });
  m.def("zeros", [](Shape& shape) { return zeros(shape, DType::kF32); });

  // Ones
  m.def("ones", [](Shape& shape, DType dtype) { return ones(shape, dtype); });
  m.def("ones", [](Shape& shape) { return ones(shape, DType::kF32); });

  // Constant
  m.def("constant", [](Shape& shape, DType dtype, float c) {
    return constant(shape, dtype, c);
  });
  m.def("constant",
        [](Shape& shape, float c) { return constant(shape, DType::kF32, c); });

  // Random
  m.def("uniform", [](Shape& shape, float lo, float hi) {
    static PRng prng;
    return uniform(prng, shape, lo, hi);
  });

  // Identity
  m.def("identity", [](Tensor& x) { return identity(x); });
  m.def("identity", [](Tensor& dst, Tensor& x) {
    dst.set_zero();
    return identity(dst, x);
  });

  // Transpose
  m.def(
      "transpose", [](Tensor& x) { return transpose(x); },
      "Transpose a tensor. Only supports reversing indices at the moment.");
  m.def(
      "transpose",
      [](Tensor& dst, Tensor& x) {
        dst.set_zero();
        return transpose(dst, x);
      },
      "Transpose a tensor and store in `dst`. Only supports reversing indices "
      "at the moment.");

  // Neg
  m.def(
      "neg", [](Tensor& x) { return neg(x); }, "Negate a tensor.");
  m.def(
      "neg",
      [](Tensor& dst, Tensor& x) {
        dst.set_zero();
        return neg(dst, x);
      },
      "Neg a tensor and store in `dst`.");

  // Exp
  m.def(
      "exp", [](Tensor& x) { return exp(x); }, "Compute e^x");
  m.def(
      "exp",
      [](Tensor& dst, Tensor& x) {
        dst.set_zero();
        return exp(dst, x);
      },
      "Compute e^x. Store into existing buffer.");

  // Log
  m.def(
      "log", [](Tensor& x) { return log(x); }, "Compute ln(x)");
  m.def(
      "log",
      [](Tensor& dst, Tensor& x) {
        dst.set_zero();
        return log(dst, x);
      },
      "Compute ln(x). Store into existing buffer.");

  // Rcp
  m.def(
      "rcp", [](Tensor& x) { return rcp(x); }, "Compute 1 / x");
  m.def(
      "rcp",
      [](Tensor& dst, Tensor& x) {
        dst.set_zero();
        return rcp(dst, x);
      },
      "Compute 1 / x. Store into existing buffer.");

  // Max
  m.def(
      "max", [](Tensor& x, float c) { return max(x, c); }, "Compute max(x, c)");
  m.def(
      "max",
      [](Tensor& dst, Tensor& x, float c) {
        dst.set_zero();
        return max(dst, x, c);
      },
      "Compute max(x, c). Store into existing buffer.");

  // MaskEq
  m.def(
      "mask_gt", [](Tensor& x, float c) { return mask_gt(x, c); },
      "Creates mask with 1.0 where x == c, 0.0 otherwise.");
  m.def(
      "mask_gt",
      [](Tensor& dst, Tensor& x, float c) {
        dst.set_zero();
        return mask_gt(dst, x, c);
      },
      "Creates mask with 1.0 where x == c, 0.0 otherwise. Store into existing "
      "buffer.");

  // Add
  m.def(
      "add", [](Tensor& x, Tensor& y) { return add(x, y); },
      "Add two tensors.");
  m.def(
      "add",
      [](Tensor& dst, Tensor& x, Tensor& y) {
        dst.set_zero();
        return add(dst, x, y);
      },
      "Add two tensors into existing buffer.");

  // Sub
  m.def(
      "sub", [](Tensor& x, Tensor& y) { return sub(x, y); },
      "Subtract two tensors.");
  m.def(
      "sub",
      [](Tensor& dst, Tensor& x, Tensor& y) {
        dst.set_zero();
        return sub(dst, x, y);
      },
      "Subtract two tensors into existing buffer.");

  // Mul
  m.def(
      "mul", [](Tensor& x, Tensor& y) { return mul(x, y); },
      "Multiply two tensors.");
  m.def(
      "mul",
      [](Tensor& dst, Tensor& x, Tensor& y) {
        dst.set_zero();
        return mul(dst, x, y);
      },
      "Multiply two tensors into existing buffer.");

  // Div
  m.def(
      "div", [](Tensor& x, Tensor& y) { return div(x, y); },
      "Divide two tensors.");
  m.def(
      "div",
      [](Tensor& dst, Tensor& x, Tensor& y) {
        dst.set_zero();
        return div(dst, x, y);
      },
      "Divide two tensors into existing buffer.");

  // Matmul
  m.def(
      "matmul", [](Tensor& x, Tensor& y) { return mmul(x, y); },
      "Matrix Multiply two tensors.");
  m.def(
      "matmul",
      [](Tensor& dst, Tensor& x, Tensor& y) {
        dst.set_zero();
        return mmul(dst, x, y);
      },
      "Matrix Multiply two tensors into existing buffer.");

  // Reduce Sum
  auto check_reduce_args = [](py::args args) {
    if (args.size() != 1) {
      throw py::value_error("Expected one argument: axis.");
    }

    if (!py::isinstance<py::int_>(args[0])) {
      throw py::type_error("Expected `axis` to be an int.");
    }
  };
  m.def(
      "reduce_sum",
      [&check_reduce_args](Tensor& dst, Tensor& x, py::args args) {
        check_reduce_args(args);
        dst.set_zero();
        return reduce_sum(dst, x, args[0].cast<int>());
      },
      "Reduce sum along single axis. Store result in `dst`.");
  m.def(
      "reduce_sum",
      [&check_reduce_args](Tensor& x, py::args args) {
        check_reduce_args(args);
        return reduce_sum(x, args[0].cast<int>());
      },
      "Reduce sum along single axis.");

  // Reduce Mean
  m.def(
      "reduce_mean",
      [&check_reduce_args](Tensor& dst, Tensor& x, py::args args) {
        check_reduce_args(args);
        dst.set_zero();
        return reduce_mean(dst, x, args[0].cast<int>());
      },
      "Reduce sum along single axis. Store result in `dst`.");
  m.def(
      "reduce_mean",
      [&check_reduce_args](Tensor& x, py::args args) {
        check_reduce_args(args);
        return reduce_mean(x, args[0].cast<int>());
      },
      "Reduce sum along single axis.");

  // Broadcast
  auto check_bcast_args = [](Tensor& x, py::args args) {
    if (args.size() != 2) {
      throw py::value_error("Expected two arguments: `axis` and `size`.");
    }

    if (!py::isinstance<py::int_>(args[0])) {
      throw py::type_error("Expected `axis` to be an int.");
    }

    if (!py::isinstance<py::int_>(args[1])) {
      throw py::type_error("Expected `size` to be an int.");
    }
  };
  m.def(
      "broadcast",
      [&check_bcast_args](Tensor& dst, Tensor& x, py::args args) {
        check_bcast_args(x, args);
        dst.set_zero();
        return broadcast(dst, x, args[0].cast<int>(), args[1].cast<int>());
      },
      "Broadcast along single axis. Store result in `dst`.");
  m.def(
      "broadcast",
      [&check_bcast_args](Tensor& x, py::args args) {
        check_bcast_args(x, args);
        return broadcast(x, args[0].cast<int>(), args[1].cast<int>());
      },
      "Broadcast along single axis.");

  // Testing
  py::module_ testing = m.def_submodule("testing");
  testing.def("allclose", &peachygrad::testing::allclose,
              "Checks whether two float arrays are close.");
}
