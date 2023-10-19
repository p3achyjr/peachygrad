#pragma once
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "base.h"

namespace peachygrad {

enum class DType : uint16_t {
  kF32 = 0,
  kI32 = 1,
};
static constexpr size_t kNumDTypes = 2;

inline size_t DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kF32:
    case DType::kI32:
      return 4;
    default:
      return 0;
  }
}

std::ostream& operator<<(std::ostream& os, DType dtype);

struct Shape {
  static constexpr int kMaxDims = 16;
  size_t num_dims;
  std::array<size_t, kMaxDims> shape;

  struct Iterator {
    size_t index;
    const Shape& shape;
    Iterator(int index, const Shape& shape) : index(index), shape(shape) {}
    size_t operator*() { return shape[index]; }
    Iterator& operator++() {
      ++index;
      return *this;
    }

    bool operator==(const Iterator& other) { return index == other.index; }
    bool operator!=(const Iterator& other) { return index != other.index; }
  };

  Shape() : num_dims(0) {}
  Shape(size_t ndims) {
    if (ndims > kMaxDims) {
      ABORT("More than 16 dims. Requesting (%zu)", ndims);
    }
    num_dims = ndims;
  }
  Shape(std::initializer_list<size_t> dims) : Shape(dims.size()) {
    std::copy(dims.begin(), dims.end(), shape.begin());
  }

  Iterator begin() const { return Iterator(0, *this); }
  Iterator end() const { return Iterator(num_dims, *this); }

  Shape transpose() const {
    Shape shape_t(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      shape_t[num_dims - i - 1] = shape[i];
    }

    return shape_t;
  }

  static Shape Zero(size_t num_dims) {
    Shape zero_index(num_dims);
    std::fill(zero_index.shape.begin(), zero_index.shape.end(), 0);
    return zero_index;
  }

  size_t operator[](size_t i) const { return shape[i]; };
  size_t& operator[](size_t i) { return shape[i]; }
  bool operator==(const Shape& other) const {
    if (!(num_dims == other.num_dims)) return false;
    for (int i = 0; i < num_dims; ++i) {
      if (shape[i] != other[i]) return false;
    }

    return true;
  }

  bool operator!=(const Shape& other) const { return !(*this == other); }
};

std::ostream& operator<<(std::ostream& os, const Shape& s);

// Offset into a tensor stored in row-major order.
size_t LinOffset(const Shape& index, const Shape& shape);

class TensorBuf final {
 public:
  TensorBuf(DType dtype, Shape shape);
  ~TensorBuf();

  inline void* raw() const { return buf_; }
  inline size_t nelems() const { return nelems_; }

 private:
  const DType dtype_;
  const Shape shape_;
  size_t nbytes_;
  size_t nelems_;
  std::array<size_t, Shape::kMaxDims> sub_elem_counts_;
  void* buf_;
};

/*
 * An n-dimensional tensor.
 */
class Tensor final {
 public:
  Tensor(DType dtype, std::initializer_list<size_t> dims);
  Tensor(DType dtype, const Shape& shape);

  inline TensorBuf& buf() { return *tensor_buf_; }
  inline size_t nelems() const { return tensor_buf_->nelems(); }
  inline void* raw() const { return tensor_buf_->raw(); }
  inline DType dtype() const { return dtype_; }
  inline const Shape& shape() const { return shape_; }
  inline bool is_vec() const { return shape_.num_dims == 1; }
  inline bool is_matrix() const { return shape_.num_dims == 2; }
  inline bool has_at_least_ndims(size_t n) { return shape_.num_dims <= n; }

  bool operator==(const Tensor& other);

 private:
  const DType dtype_;
  Shape shape_;
  std::shared_ptr<TensorBuf> tensor_buf_;
};

// FnType is a closure that takes the current index and the value at that
// index.
template <typename InitFnType, typename FnType, typename EndFnType>
void GenericIterate(const Tensor& t, InitFnType init_fn, FnType fn,
                    EndFnType end_fn) {
  auto is_lt = [](const Shape& lhs, const Shape& rhs) {
    if (rhs.num_dims > lhs.num_dims) return false;
    if (rhs.num_dims < lhs.num_dims) return true;
    for (int i = 0; i < lhs.num_dims; ++i) {
      if (lhs[i] < rhs[i]) {
        return true;
      } else if (lhs[i] > rhs[i]) {
        return false;
      }
    }

    // equal.
    return false;
  };

  auto is_gt = [](const Shape& lhs, const Shape& rhs) {
    if (rhs.num_dims < lhs.num_dims) return false;
    if (rhs.num_dims > lhs.num_dims) return true;
    for (int i = 0; i < lhs.num_dims; ++i) {
      if (lhs[i] > rhs[i]) {
        return true;
      } else if (lhs[i] < rhs[i]) {
        return false;
      }
    }

    // equal.
    return false;
  };

  auto inc = [](Shape& index, const Shape& shape) {
    // these should have the same ndims if we are localizing to this fn.
    size_t carry_bit = 0;
    for (int i = index.num_dims - 1; i >= 0; --i) {
      if (i == index.num_dims - 1) {
        index[i]++;
      }

      index[i] += carry_bit;
      if (i > 0) index[i] %= shape[i];
      carry_bit = index[i] == 0 ? 1 : 0;

      if (!carry_bit) {
        break;
      }
    }
  };

  init_fn();
  Shape index(Shape::Zero(t.shape().num_dims));
  Shape ub(Shape::Zero(t.shape().num_dims));
  ub[0] = t.shape()[0];

  for (; is_lt(index, ub); inc(index, t.shape())) {
    size_t lin_offset = LinOffset(index, t.shape());
    char* buf = static_cast<char*>(t.raw()) + DTypeSize(t.dtype()) * lin_offset;
    fn(index, static_cast<void*>(buf));
  }
  end_fn();
}

std::ostream& operator<<(std::ostream& os, const Tensor& s);

}  // namespace peachygrad
