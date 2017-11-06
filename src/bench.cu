// -*- compile-command: "nvcc -D__STRICT_ANSI__ -ccbin clang-3.8 -Xcompiler -Wall,-Wextra -g -G -lineinfo -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <algorithm>
#include <cstdlib>

using namespace std;

#define constexpr

template< size_t N >
struct myarray {
    typedef uint32_t value_type;

    value_type data[N];

    myarray() { }

    // Should be constexpr
    __host__ __device__
    const value_type &operator[](size_t i) const {
        return data[i];
    }

    __host__ __device__
    value_type &operator[](size_t i) {
        return data[i];
    }
};


template< size_t N >
struct myfunc {
    __device__
    typename myarray<N>::value_type
    operator()(const myarray<N> &arr) {
        typename myarray<N>::value_type sum = 0;
        for (size_t i = 0; i < N; ++i)
            sum += arr[i];
        return sum;
    }
};

template< size_t N >
struct mysum {
    __device__
    typename myarray<N>
    operator()(const myarray<N> &u, const myarray<N> &v) {
        typename myarray<N>::value_type sum = 0;
        for (size_t i = 0; i < N; ++i)
            sum += arr[i];
        return sum;
    }
};


template< int MOD, size_t N >
struct myrand {
    // Candidate for a "move" operator?
    myarray<N> operator()() {
        myarray<N> arr;
        for (int i = 0; i < N; ++i)
            arr[i] = (typename myarray<N>::value_type) (rand() % MOD);
        return arr;
    }
};


template< size_t N >
ostream &
operator<<(ostream &os, const myarray<N> &arr) {
    os << "( " << arr[0];
    for (int i = 1; i < N; ++i)
        os << ", " << arr[i];
    os << " )" << flush;
    return os;
}


template< typename T >
ostream &
operator<<(ostream &os, const thrust::host_vector<T> &v) {
    os << "vector has size " << v.size() << endl;

    for(int i = 0; i < v.size(); i++)
        os << "v[" << i << "] = " << v[i] << endl;
    return os;
}


#define NELTS 7

int main(int argc, char *argv[]) {
    long n = 16;
    if (argc > 1)
        n = atol(argv[1]);

//    constexpr size_t NELTS = 7;
    typedef myarray<NELTS> array;
    typedef typename array::value_type value_type;

    // generate n random numbers serially
    thrust::host_vector< array > h_vec(n);
    ::generate(h_vec.begin(), h_vec.end(), myrand<32, NELTS>());

    cout << "h_vec = " << h_vec << endl;

    // transfer data to the device
    thrust::device_vector< array > d_vec = h_vec;
    thrust::device_vector< value_type > d_res(n);

    thrust::transform(d_vec.begin(), d_vec.end(), d_res.begin(), myfunc<NELTS>());

    // transfer data back to host
    thrust::host_vector<value_type> res = d_res;
    //thrust::copy(d_vec.begin(), d_vec.end(), res.begin());

    cout << "res = " << res << endl;
    return 0;
}
