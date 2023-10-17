#include "tensor.h"

namespace peachygrad {
namespace {

struct TensorStringBuilder {
  const Tensor& tensor;
  std::stringstream ss;
  Shape last_index;

  TensorStringBuilder(const Tensor& tensor)
      : tensor(tensor), last_index(Shape::Zero(tensor.shape().num_dims)) {}

  void init_fn() {
    for (int i = 0; i < last_index.num_dims; ++i) {
      ss << "[";
    }
  }

  void end_fn() {
    for (int i = 0; i < last_index.num_dims; ++i) {
      ss << "]";
    }
  }

  void fn(const Shape& index, void* it) {
    int num_carry = 0;
    for (int i = 0; i < index.num_dims; ++i) {
      if (index[i] == 0 && last_index[i] != 0) ++num_carry;
    }

    // must be contiguous, i.e. [0, 1, 0, 1] -> [1, 0, 0, 0] is impossible.
    if (num_carry > 0) {
      for (int i = 0; i < num_carry; ++i) {
        ss << "]";
      }
      ss << "\n";
      if (num_carry > 1) ss << "\n";

      // handle nesting depth.
      for (int i = 0; i < index.num_dims - num_carry; ++i) ss << " ";
      for (int i = 0; i < num_carry; ++i) {
        ss << "[";
      }
    }

    if (tensor.dtype() == DType::kF32) {
      ss << *(float*)(it) << " ";
    } else {
      ss << *(int*)(it) << " ";
    }

    last_index = index;
  }

  std::string build() {
    GenericIterate(
        tensor, [this]() { init_fn(); },
        [this](const Shape& index, void* it) { fn(index, it); },
        [this]() { end_fn(); });
    return ss.str();
  }
};

template <typename T>
bool buf_eq(T* buf, T* other_buf, size_t n) {
  for (int i = 0; i < n; ++i) {
    if (buf[i] != other_buf[i]) return false;
  }

  return true;
}
}  // namespace

std::ostream& operator<<(std::ostream& os, DType dtype) {
  switch (dtype) {
    case DType::kF32:
      os << "f32";
      break;
    case DType::kI32:
      os << "i32";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Shape& s) {
  os << "(";
  for (int i = 0; i < s.num_dims - 1; ++i) os << s.shape[i] << ", ";
  os << s.shape[s.num_dims - 1] << ")";
  return os;
}

size_t LinOffset(const Shape& index, const Shape& shape) {
  std::array<size_t, Shape::kMaxDims> sub_sizes;
  size_t sub_size = 1;
  for (int i = shape.num_dims - 1; i >= 0; --i) {
    sub_sizes[i] = sub_size;
    sub_size *= shape[i];
  }

  size_t lin_offset = 0;
  for (int i = 0; i < shape.num_dims; ++i) {
    lin_offset += index[i] * sub_sizes[i];
  }

  return lin_offset;
}

//---- TensorBuf ----//
TensorBuf::TensorBuf(DType dtype, Shape shape) : dtype_(dtype), shape_(shape) {
  size_t dtype_size = DTypeSize(dtype);
  size_t nelems = 1;
  for (int i = shape_.num_dims - 1; i >= 0; --i) {
    sub_elem_counts_[i] = nelems;
    nelems *= shape_[i];
  }

  nelems_ = nelems;
  nbytes_ = nelems * dtype_size;

  buf_ = aligned_alloc(MM_ALIGN, nbytes_);
  memset(buf_, 0, nbytes_);
}

TensorBuf::~TensorBuf() { free(buf_); }

//---- Tensor ----//
Tensor::Tensor(DType dtype, std::initializer_list<size_t> dims)
    : dtype_(dtype),
      shape_(dims),
      tensor_buf_(std::make_shared<TensorBuf>(dtype_, shape_)) {}
Tensor::Tensor(DType dtype, const Shape& shape)
    : dtype_(dtype),
      shape_(shape),
      tensor_buf_(std::make_shared<TensorBuf>(dtype_, shape_)) {}

bool Tensor::operator==(const Tensor& other) {
  if (dtype_ != other.dtype()) return false;
  if (shape_ != other.shape()) return false;

  size_t num_elems = tensor_buf_->nelems();
  if (dtype_ == DType::kF32) {
    return buf_eq<float>(static_cast<float*>(raw()),
                         static_cast<float*>(other.raw()), num_elems);
  } else {
    return buf_eq<int>(static_cast<int*>(raw()), static_cast<int*>(other.raw()),
                       num_elems);
  }
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  TensorStringBuilder builder(t);
  os << builder.build();
  os << ", dtype: " << t.dtype() << ", shape: " << t.shape();
  return os;
}

}  // namespace peachygrad
