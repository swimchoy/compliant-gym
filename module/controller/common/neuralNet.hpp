//
// Created by suyoung on 21. 8. 2..
//

#ifndef NEURAL_NETWORK_HPP_
#define NEURAL_NETWORK_HPP_

namespace raisim
{

namespace nn
{

enum class ActivationType {
  linear,
  relu,
  leaky_relu,
  tanh,
  sigmoid,
  softsign
};

template<typename dtype, ActivationType activationType>
struct Activation {
  inline void nonlinearity(Eigen::Matrix<dtype, -1, -1> &output) {}
  inline void nonlinearity(Eigen::Matrix<dtype, -1, 1> &output) {}
};

template<typename dtype>
struct Activation<dtype, ActivationType::relu> {
  inline void nonlinearity(Eigen::Matrix<dtype, -1, -1> &output) { output = output.cwiseMax(0.); }
  inline void nonlinearity(Eigen::Matrix<dtype, -1, 1> &output) { output = output.cwiseMax(0.); }
};

template<typename dtype>
struct Activation<dtype, ActivationType::leaky_relu> {
  inline void nonlinearity(Eigen::Matrix<dtype, -1, -1> &output) { output = output.cwiseMax(1.e-2 * output); }
  inline void nonlinearity(Eigen::Matrix<dtype, -1, 1> &output) { output = output.cwiseMax(1.e-2 * output); }
};

template<typename dtype>
struct Activation<dtype, ActivationType::sigmoid> {
  inline void nonlinearity(Eigen::Matrix<dtype, -1, -1> &output) { output = ((-output).array().exp() + 1.).cwiseInverse(); }
  inline void nonlinearity(Eigen::Matrix<dtype, -1, 1> &output) { output = ((-output).array().exp() + 1.).cwiseInverse(); }
};

template<typename dtype>
struct Activation<dtype, ActivationType::tanh> {
  inline void nonlinearity(Eigen::Matrix<dtype, -1, -1> &output) { output = output.array().tanh(); }
  inline void nonlinearity(Eigen::Matrix<dtype, -1, 1> &output) { output = output.array().tanh(); }
  inline Eigen::Matrix<dtype, -1, 1> _nonlinearity(const Eigen::Matrix<dtype, -1, 1> &output) {
    Eigen::Matrix<dtype, -1, 1> out = output;
    return out.array().tanh();
  }
};

template<typename dtype>
struct Activation<dtype, ActivationType::softsign> {
  inline void nonlinearity(Eigen::Matrix<dtype, -1, -1> &output) {
    for (int i = 0; i < output.size(); ++i)
      output[i] = output[i] / (std::abs(output[i]) + 1.0);
  }
  inline void nonlinearity(Eigen::Matrix<dtype, -1, 1> &output) {
    for (int i = 0; i < output.size(); ++i)
      output[i] = output[i] / (std::abs(output[i]) + 1.0);
  }
};

template<typename dtype, int inputDim, int outputDim, ActivationType activationType>
class Linear {
 public:
  typedef Eigen::Matrix<dtype, outputDim, 1> Output;
  typedef Eigen::Matrix<dtype, inputDim, 1> Input;

  Linear(std::vector<int> hiddenSize) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    architecture_.push_back(inputDim);
    architecture_.reserve(architecture_.size() + hiddenSize.size());
    architecture_.insert(architecture_.end(), hiddenSize.begin(), hiddenSize.end());
    architecture_.push_back(outputDim);

    params.resize(2 * (architecture_.size() - 1));
    W.resize(architecture_.size() - 1);
    b.resize(architecture_.size() - 1);
    x.resize(architecture_.size());

    for (int i = 0; i < (int)params.size(); ++i) {
      if (i % 2 == 0) {
        W[i / 2].resize(architecture_[i / 2 + 1], architecture_[i / 2]);
        params[i].resize(architecture_[i / 2] * architecture_[i / 2 + 1]);
      }
      else if (i % 2 == 1) {
        b[(i-1) / 2].resize(architecture_[(i+1) / 2]);
        params[i].resize(architecture_[(i+1) / 2]);
      }
    }
  }

  void readParamFromTxt(std::string filePath) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;

    int totalN = 0;
    /// assign parameters
    for (int i = 0; i < int(params.size()); ++i) {
      int paramSize = 0;

      while (std::getline(lineStream, cell, ',')) {
        params[i](paramSize++) = std::stod(cell);
        if (paramSize == params[i].size()) break;
      }
      totalN += paramSize;
      if (i % 2 == 0)
        memcpy(W[i / 2].data(), params[i].data(), sizeof(dtype) * W[i / 2].size());
      if (i % 2 == 1)
        memcpy(b[i / 2].data(), params[i].data(), sizeof(dtype) * b[(i - 1) / 2].size());
    }
  }

  inline Output forward(Input &input) {
    x[0] = input;
    for (int i = 0; i < (int)W.size() - 1; ++i) {
      x[i + 1] = W[i] * x[i] + b[i];
      activation_.nonlinearity(x[i + 1]);
    }
    x[x.size() - 1] = W[W.size() - 1] * x[x.size() - 2] + b[b.size() - 1];

    return x.back();
  }

 private:
  std::vector<Eigen::Matrix<dtype, -1, 1>> params;
  std::vector<Eigen::Matrix<dtype, -1, -1>> W;
  std::vector<Eigen::Matrix<dtype, -1, 1>> b;
  std::vector<Eigen::Matrix<dtype, -1, 1>> x;

  Activation<dtype, activationType> activation_;
  std::vector<int> architecture_;
};

template<typename dtype, int inputDim, int hiddenDim>
class LSTM {
 public:
  typedef Eigen::Matrix<dtype, hiddenDim, 1> Output;
  typedef Eigen::Matrix<dtype, inputDim, 1> Input;

  LSTM(int numLayer) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    numLayer_ = numLayer;
    initHidden();

    architecture_.push_back(inputDim);
    for (int l = 0; l < numLayer_; ++l) {
      architecture_.push_back(hiddenDim);
      architecture_.push_back(hiddenDim);
    }

    params.resize(2 * (architecture_.size() - 1));
    W.resize(architecture_.size() - 1);
    b.resize(architecture_.size() - 1);
    x.resize(architecture_.size());

    for (int w = 0; w < params.size(); ++w) {
      if (w % 4 == 0) {
        if (w / 4 == 0){ /// first layer
          W[w/2].resize(4 * hiddenDim, inputDim);
          W[w/2 + 1].resize(4 * hiddenDim, hiddenDim);
          b[w/2].resize(4 * hiddenDim);
          b[w/2 + 1].resize(4 * hiddenDim);

          params[w + 0].resize(4 * inputDim * hiddenDim);
          params[w + 1].resize(4 * hiddenDim * hiddenDim);
          params[w + 2].resize(4 * hiddenDim);
          params[w + 3].resize(4 * hiddenDim);
        }
        else { /// not the first layer
          W[w/2].resize(4 * hiddenDim, hiddenDim);
          W[w/2 + 1].resize(4 * hiddenDim, hiddenDim);
          b[w/2].resize(4 * hiddenDim);
          b[w/2 + 1].resize(4 * hiddenDim);

          params[w + 0].resize(4 * hiddenDim * hiddenDim);
          params[w + 1].resize(4 * hiddenDim * hiddenDim);
          params[w + 2].resize(4 * hiddenDim);
          params[w + 3].resize(4 * hiddenDim);
        }
        w += 3;
      }
    }
  }

  void initHidden() {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    h.resize(numLayer_);
    c.resize(numLayer_);

    for (int i = 0; i < numLayer_; ++i) {
      h[i].setZero(hiddenDim);
      c[i].setZero(hiddenDim);
    }
  }
  
  void readParamFromTxt(std::string filePath) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    
    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;
    
    int totalN = 0;

    for (int i = 0; i < int(params.size()); ++i) {
      int paramSize = 0;
      
      while (std::getline(lineStream, cell, ',')) {
        params[i](paramSize++) = std::stod(cell);
        if (paramSize == params[i].size()) break;
      }
      totalN += paramSize;
    }
    for (int i = 0; i < int(params.size()); ++i) {
      memcpy(W[i/2 + 0].data(), params[i + 0].data(), sizeof(dtype) * W[i/2 + 0].size());
      memcpy(W[i/2 + 1].data(), params[i + 1].data(), sizeof(dtype) * W[i/2 + 1].size());
      memcpy(b[i/2 + 0].data(), params[i + 2].data(), sizeof(dtype) * b[i/2 + 0].size());
      memcpy(b[i/2 + 1].data(), params[i + 3].data(), sizeof(dtype) * b[i/2 + 1].size());
      i += 3;
    }
  }
  
  inline Output forward(Input &input) {
    x[0] = input;

    /// lstm
    for (int l = 0; l < numLayer_; ++l) {
      x[2*l + 1] = W[2*l] * x[2*l] + b[2*l] + W[2*l + 1] * h[l] + b[2*l + 1];

      Eigen::Matrix<dtype, -1, 1> i = x[2*l + 1].segment(0*hiddenDim, hiddenDim),
                                  f = x[2*l + 1].segment(1*hiddenDim, hiddenDim),
                                  g = x[2*l + 1].segment(2*hiddenDim, hiddenDim),
                                  o = x[2*l + 1].segment(3*hiddenDim, hiddenDim);

      sigmoid_.nonlinearity(i);
      sigmoid_.nonlinearity(f);
      tanh_.nonlinearity(g);
      sigmoid_.nonlinearity(o);

      c[l] = f.cwiseProduct(c[l]) + i.cwiseProduct(g);
      h[l] = o.cwiseProduct(tanh_._nonlinearity(c[l]));
      x[2*l + 2] = h[l];
    }

    return x.back();
  }

 private:
  int numLayer_;
  std::vector<Eigen::Matrix<dtype, -1, 1>> h;
  std::vector<Eigen::Matrix<dtype, -1, 1>> c;

  std::vector<Eigen::Matrix<dtype, -1, 1>> params;
  std::vector<Eigen::Matrix<dtype, -1, -1>> W;
  std::vector<Eigen::Matrix<dtype, -1, 1>> b;
  std::vector<Eigen::Matrix<dtype, -1, 1>> x;

  Activation<dtype, ActivationType::sigmoid> sigmoid_;
  Activation<dtype, ActivationType::tanh> tanh_;
  std::vector<int> architecture_;
};

template<typename dtype, int inputDim, int outputDim, ActivationType activationType>
class LSTM_MLP {
 public:
  typedef Eigen::Matrix<dtype, outputDim, 1> Output;
  typedef Eigen::Matrix<dtype, inputDim, 1> Input;

  LSTM_MLP(int hiddenDim, int numLayer, std::vector<int> outputMLP) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    hiddenDim_ = hiddenDim;
    numLayer_ = numLayer;
    initHidden();

    architecture_.push_back(inputDim);
    for (int l = 0; l < numLayer_; ++l) {
      architecture_.push_back(hiddenDim_);
      architecture_.push_back(hiddenDim_);
    }
    architecture_.reserve(architecture_.size() + outputMLP.size());
    architecture_.insert(architecture_.end(), outputMLP.begin(), outputMLP.end());
    architecture_.push_back(outputDim);

    params.resize(2 * (architecture_.size() - 1));
    W.resize(architecture_.size() - 1);
    b.resize(architecture_.size() - 1);
    x.resize(architecture_.size());

    for (int w = 0; w < params.size(); ++w) {
      if (w < 4 * numLayer_) { /// lstm
        if (w % 4 == 0) {
          if (w / 4 == 0){ /// first layer
            W[w/2].resize(4 * hiddenDim_, inputDim);
            W[w/2 + 1].resize(4 * hiddenDim_, hiddenDim_);
            b[w/2].resize(4 * hiddenDim_);
            b[w/2 + 1].resize(4 * hiddenDim_);

            params[w + 0].resize(4 * inputDim * hiddenDim_);
            params[w + 1].resize(4 * hiddenDim_ * hiddenDim_);
            params[w + 2].resize(4 * hiddenDim_);
            params[w + 3].resize(4 * hiddenDim_);
          }
          else { /// not the first layer
            W[w/2].resize(4 * hiddenDim_, hiddenDim_);
            W[w/2 + 1].resize(4 * hiddenDim_, hiddenDim_);
            b[w/2].resize(4 * hiddenDim_);
            b[w/2 + 1].resize(4 * hiddenDim_);

            params[w + 0].resize(4 * hiddenDim_ * hiddenDim_);
            params[w + 1].resize(4 * hiddenDim_ * hiddenDim_);
            params[w + 2].resize(4 * hiddenDim_);
            params[w + 3].resize(4 * hiddenDim_);
          }
          w += 3;
        }
      } else { /// mlp
        if (w % 2 == 0) {
          W[w / 2].resize(architecture_[w / 2 + 1], architecture_[w / 2]);
          params[w].resize(architecture_[w / 2] * architecture_[w / 2 + 1]);
        } else if (w % 2 == 1) {
          b[(w - 1)/ 2].resize(architecture_[(w + 1) / 2]);
          params[w].resize(architecture_[(w + 1) / 2]);
        }
      }
    }
  }

  void initHidden() {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    h.resize(numLayer_);
    c.resize(numLayer_);

    for (int i = 0; i < numLayer_; ++i) {
      h[i].setZero(hiddenDim_);
      c[i].setZero(hiddenDim_);
    }
  }
  
  void readParamFromTxt(std::string filePath) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    
    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;
    
    int totalN = 0;

    for (int i = 0; i < int(params.size()); ++i) {
      int paramSize = 0;
      
      while (std::getline(lineStream, cell, ',')) {
        params[i](paramSize++) = std::stod(cell);
        if (paramSize == params[i].size()) break;
      }
      totalN += paramSize;
    }
    for (int i = 0; i < int(params.size()); ++i) {
      if (i < 4 * numLayer_) {
        memcpy(W[i/2 + 0].data(), params[i + 0].data(), sizeof(dtype) * W[i/2 + 0].size());
        memcpy(W[i/2 + 1].data(), params[i + 1].data(), sizeof(dtype) * W[i/2 + 1].size());
        memcpy(b[i/2 + 0].data(), params[i + 2].data(), sizeof(dtype) * b[i/2 + 0].size());
        memcpy(b[i/2 + 1].data(), params[i + 3].data(), sizeof(dtype) * b[i/2 + 1].size());
        i += 3;
      } else {
        if (i % 2 == 0)
          memcpy(W[i / 2].data(), params[i].data(), sizeof(dtype) * W[i / 2].size());
        if (i % 2 == 1)
          memcpy(b[i / 2].data(), params[i].data(), sizeof(dtype) * b[(i - 1) / 2].size());
      }
    }
  }
  
  inline Output forward(Input &input) {
    x[0] = input;

    /// lstm
    for (int l = 0; l < numLayer_; ++l) {
      x[2*l + 1] = W[2*l] * x[2*l] + b[2*l] + W[2*l + 1] * h[l] + b[2*l + 1];

      Eigen::Matrix<dtype, -1, 1> i = x[2*l + 1].segment(0*hiddenDim_, hiddenDim_),
                                  f = x[2*l + 1].segment(1*hiddenDim_, hiddenDim_),
                                  g = x[2*l + 1].segment(2*hiddenDim_, hiddenDim_),
                                  o = x[2*l + 1].segment(3*hiddenDim_, hiddenDim_);

      sigmoid_.nonlinearity(i);
      sigmoid_.nonlinearity(f);
      tanh_.nonlinearity(g);
      sigmoid_.nonlinearity(o);

      c[l] = f.cwiseProduct(c[l]) + i.cwiseProduct(g);
      h[l] = o.cwiseProduct(tanh_._nonlinearity(c[l]));
      x[2*l + 2] = h[l];
    }

    /// mlp
    for (int i = 2*numLayer_; i < (int)x.size()-1; ++i) {
      x[i + 1] = W[i] * x[i] + b[i];
      activation_.nonlinearity(x[i + 1]);
    }
    x[x.size() - 1] = W[W.size() - 1] * x[x.size() - 2] + b[b.size() - 1];

    return x.back();
  }

 private:
  int hiddenDim_, numLayer_;
  std::vector<Eigen::Matrix<dtype, -1, 1>> h;
  std::vector<Eigen::Matrix<dtype, -1, 1>> c;

  std::vector<Eigen::Matrix<dtype, -1, 1>> params;
  std::vector<Eigen::Matrix<dtype, -1, -1>> W;
  std::vector<Eigen::Matrix<dtype, -1, 1>> b;
  std::vector<Eigen::Matrix<dtype, -1, 1>> x;

  Activation<dtype, activationType> activation_;
  Activation<dtype, ActivationType::sigmoid> sigmoid_;
  Activation<dtype, ActivationType::tanh> tanh_;
  std::vector<int> architecture_;
};

} // namespace nn

} // namespace raisim

#endif    // NEURAL_NETWORK_HPP_