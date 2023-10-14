#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <utility>

#include "api.h"
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
      .def("__repr__", [](const Shape& shape) {
        std::stringstream ss;
        ss << shape;
        return ss.str();
      });

  py::class_<Tensor>(m, "Tensor")
      .def("dtype", &Tensor::dtype)
      .def("shape", &Tensor::shape)
      .def("__repr__", [](const Tensor& tensor) {
        std::stringstream ss;
        ss << tensor;
        return ss.str();
      });

  // API.
  m.attr("f32") = DType::kF32;
  m.attr("i32") = DType::kI32;
  m.def(
      "tensor", [](py::list py_list) { return tensor(py_list, DType::kF32); },
      "Initialize tensor (default f32).");
  m.def(
      "tensor",
      [](py::list py_list, DType dtype) { return tensor(py_list, dtype); },
      "Initialize tensor.");
  m.def(
      "add", [](Tensor& x, Tensor& y) { return add(x, y); },
      "Add two tensors.");
  m.def(
      "sub", [](Tensor& x, Tensor& y) { return sub(x, y); },
      "Subtract two tensors.");
  m.def(
      "mul", [](Tensor& x, Tensor& y) { return mul(x, y); },
      "Multiply two tensors.");
  m.def(
      "div", [](Tensor& x, Tensor& y) { return div(x, y); },
      "Divide two tensors.");
  m.def(
      "matmul", [](Tensor& x, Tensor& y) { return mmul(x, y); },
      "Matrix Multiply two tensors.");
}
