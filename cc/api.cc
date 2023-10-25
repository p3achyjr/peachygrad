#include "api.h"

namespace peachygrad {
namespace {
namespace py = ::pybind11;
}

Tensor tensor(
    py::array_t<float, py::array::c_style | py::array::forcecast> np_arr) {
  py::buffer_info buf_info = np_arr.request();
  float* np_raw = static_cast<float*>(buf_info.ptr);
  int np_ndims = buf_info.ndim;
  std::vector<ssize_t> np_shape = buf_info.shape;

  // Copy.
  Shape shape(np_ndims);
  for (int i = 0; i < np_shape.size(); ++i) {
    shape[i] = np_shape[i];
  }
  Tensor t(DType::kF32, shape);
  std::copy(np_raw, np_raw + buf_info.size, static_cast<float*>(t.raw()));
  return t;
}

Tensor tensor(py::list py_list, DType dtype) {
  Shape shape;
  py::list clist = py_list;
  while (true) {
    shape[shape.num_dims] = clist.size();
    shape.num_dims++;
    if (clist.size() == 0 || !py::isinstance<py::list>(clist[0])) {
      break;
    }

    clist = clist[0].cast<py::list>();
  }

  Tensor t(dtype, shape);
  auto set_index = [py_list, dtype](const Shape& index, void* it) {
    // find element in list.
    py::object obj = py_list;
    for (size_t i : index) {
      obj = obj.cast<py::list>()[i];
    }

    if (dtype == DType::kI32) {
      int val = obj.cast<int>();
      *((int*)it) = val;
    } else {
      float val = obj.cast<float>();
      *((float*)it) = val;
    }
  };

  GenericIterate(
      t, []() {}, set_index, []() {});
  return t;
}

py::array_t<float> numpy(Tensor& t) {
  void* data = t.raw();

  std::vector<ssize_t> np_shape;
  for (auto d : t.shape()) {
    np_shape.emplace_back(d);
  }

  py::array_t<float> np_array(np_shape);
  memcpy(np_array.mutable_data(), data, t.nelems() * sizeof(float));
  return np_array;
}

namespace testing {
namespace {
template <typename T>
float get_atol(T ref, T x) {
  return abs(ref - x);
}

template <typename T>
float get_rtol(T ref, T x) {
  return abs(ref - x) / std::max(abs(x), abs(ref));
}
}  // namespace

bool allclose(Tensor& ref, Tensor& x, float atol, float rtol) {
  float max_rtol = 0.0f;
  float max_atol = 0.0f;
  Shape max_rtol_shape;
  Shape max_atol_shape;
  DType dtype = x.dtype();

  auto cmp = [&](const Shape& index, void* it) {
    float this_x = *((float*)it);
    float this_ref = *((float*)ref.raw() + LinOffset(index, ref.shape()));
    float this_atol = get_atol(this_ref, this_x);
    float this_rtol = get_rtol(this_ref, this_x);

    if (this_atol > max_atol) {
      max_atol = this_atol;
      max_atol_shape = index;
    }

    if (this_rtol > max_rtol) {
      max_rtol = this_rtol;
      max_rtol_shape = index;
    }
  };

  GenericIterate(
      x, []() {}, cmp, []() {});

  if (max_rtol > rtol) {
    float ref_val =
        *((float*)ref.raw() + LinOffset(max_rtol_shape, ref.shape()));
    float x_val = *((float*)x.raw() + LinOffset(max_rtol_shape, x.shape()));
    std::cout << "Max RTol Exceeded. Max: " << rtol << ", Got: " << max_rtol
              << "\n";
    std::cout << "Index: " << max_rtol_shape << "\n";
    std::cout << "Ref: " << ref_val << "\n";
    std::cout << "x:   " << x_val << "\n";
    return false;
  }

  if (max_atol > atol) {
    float ref_val =
        *((float*)ref.raw() + LinOffset(max_atol_shape, ref.shape()));
    float x_val = *((float*)x.raw() + LinOffset(max_atol_shape, x.shape()));
    std::cout << "Max ATol Exceeded. Max: " << atol << ", Got: " << max_atol
              << "\n";
    std::cout << "Index: " << max_atol_shape << "\n";
    std::cout << "Ref: " << ref_val << "\n";
    std::cout << "x:   " << x_val << "\n";
    return false;
  }

  return true;
}

}  // namespace testing
}  // namespace peachygrad
