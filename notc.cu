#include "cute/tensor.hpp"

#include "task.h"
#include "utils.h"

using namespace cute;

using TileShapeM = _64;
using TileShapeN = _32;
using TileShapeK = _32;

using TileShape = Shape<TileShapeM, TileShapeN, TileShapeK>;

using SmemLayoutAW0 = decltype(make_layout(make_shape(TileShapeM {}, TileShapeK {})));
using SmemLayoutW1 = decltype(make_layout(make_shape(TileShapeN {}, TileShapeK {})));
using SmemLayoutAW1 = decltype(make_layout(make_shape(TileShapeM {}, TileShapeN {})));
// using SmemLayoutAW1 = decltype(make_layout(make_shape(TileShapeM {}, TileShapeN {})));
using SmemLayoutW2 = decltype(make_layout(make_shape(TileShapeN {}, TileShapeK {})));

struct SharedStorage {
    union {
        cute::array_aligned<ElementA, cosize_v<SmemLayoutAW0>> AW0;
        cute::array_aligned<ElementA, cosize_v<SmemLayoutAW1>> AW1;
    };

    union {
        cute::array_aligned<ElementB, cosize_v<SmemLayoutW1>> W1;
        cute::array_aligned<ElementB, cosize_v<SmemLayoutW2>> W2;
    };
};

template <typename MAW0, typename MW1, typename MW2, typename MAW1, typename MAW1W2, typename TA, typename TB, typename TC>
__global__ void notcDeviceKernel(
    MAW0 mAW0_mk0, MW1 mW1_k1k0, MW2 mW2_k2k1, MAW1 mAW1_mk1, MAW1W2 mAW2_mk2,
    TA tA, TB tB, TC tC)
{
    using X = Underscore;

    extern __shared__ char smemBuf[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smemBuf);

    auto sAW0 = make_tensor(make_smem_ptr(storage.AW0.data()), SmemLayoutAW0 {}); // (BLK_M,BLK_K)
    auto sW1 = make_tensor(make_smem_ptr(storage.W1.data()), SmemLayoutW1 {}); // (BLK_N,BLK_K)

    // Represent the full tensors
    // auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA); // (M,K)
    // auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB); // (N,K)
    // auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC); // (M,N)

    // Get the appropriate blocks for this thread block --
    // potential for thread block locality
    // auto blk_shape = make_shape(size<0>(sAW0), size<0>(sW1), size<1>(sW1)); // (BLK_M,BLK_N,BLK_K)
    auto blk_shape = TileShape {};
    auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)

    auto gA = local_tile(mAW0_mk0, blk_shape, blk_coord, Step<_1, X, _1> {}); // (BLK_M,BLK_K,k)
    auto gB = local_tile(mW1_k1k0, blk_shape, blk_coord, Step<X, _1, _1> {}); // (BLK_N,BLK_K,k)
    auto gC = local_tile(mAW1_mk1, blk_shape, blk_coord, Step<_1, _1, X> {}); // (BLK_M,BLK_N)

    auto tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
    auto tAsA = local_partition(sAW0, tA, threadIdx.x); // (THR_M,THR_K)

    auto tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
    auto tBsB = local_partition(sW1, tB, threadIdx.x); // (THR_N,THR_K)

    //
    // Define C accumulators and A/B partitioning
    //
    // Partition sAW0 (M,K) by the rows of tC

    auto tCsA = local_partition(sAW0, tC, threadIdx.x, Step<_1, X> {}); // (THR_M,BLK_K)
    // Partition sW1 (N,K) by the cols of tC
    auto tCsB = local_partition(sW1, tC, threadIdx.x, Step<X, _1> {}); // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1> {}); // (THR_M,THR_N)

    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC); // (THR_M,THR_N)

    // Clear the accumulators
    clear(tCrC);

    auto k_max = size<2>(tAgA);

    FIRST_THREAD_ONLY({
        println("k_max: ", k_max);
    });

    for (int k = 0; k < k_max; ++k) {
        // Copy gmem to smem
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);

        __syncthreads();

        FIRST_THREAD_ONLY({
            println("#", k, " tCsA: ", tCsA);
            println("tCsA: ", Dump(tCsA));
            println("#", k, " tCsB: ", tCsB);
            println("tCsB: ", Dump(tCsB));
        });

        // Compute gemm on smem
        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }

    FIRST_THREAD_ONLY({
        println("tCrC: ", tCrC);
        println("tCrC: ", Dump(tCrC));

        println("tCgC: ", tCgC);
        println("tCgC: ", Dump(tCgC));

        println("tCsA: ", tCsA);
        println("tCsA: ", Dump(tCsA));
    });

    copy(tCrC, tCgC);
}

void notcKernel(cudaStream_t stream, const Task& task)
{
    auto mA = make_tensor(make_gmem_ptr(task.devA), make_shape(task.m, task.k0), GenRowMajor {});

    auto mB = make_tensor(make_gmem_ptr(task.devB), make_shape(task.k1, task.k0), GenColMajor {});

    auto mC = make_tensor(make_gmem_ptr(task.devC), make_shape(task.n, task.k1), GenColMajor {});

    auto mAB = make_tensor(make_gmem_ptr(task.devAB), make_shape(task.m, task.k1), GenRowMajor {});

    auto mABC = make_tensor(make_gmem_ptr(task.devABC), make_shape(task.m, task.n), GenRowMajor {});

    dim3 const grid = dim3(
        cute::size(cute::ceil_div(task.m, cute::shape<0>(TileShape {}))),
        cute::size(cute::ceil_div(std::max(task.k1, task.n), cute::shape<1>(TileShape {}))),
        1);

    // auto tA = make_layout(make_shape(Int<32> {}, Int<8> {})); // 32 x 8
    // auto tB = make_layout(make_shape(Int<32> {}, Int<8> {})); // 32 x 8
    // auto tC = make_layout(make_shape(Int<16> {}, Int<16> {})); // 16 x 16

    auto tA = make_layout(make_shape(Int<32> {}, Int<8> {})); // 32 x 8
    auto tB = make_layout(make_shape(Int<32> {}, Int<8> {})); // 32 x 8
    auto tC = make_layout(make_shape(Int<16> {}, Int<16> {})); // 16 x 16

    dim3 const block = dim3(size(tC), 1, 1);

    int smem_size = sizeof(SharedStorage);

    println("grid: ", grid);
    println("block: ", block);
    println("smem_size: ", smem_size);

    notcDeviceKernel<<<grid, block, smem_size, stream>>>(
        mA, mB, mC, mAB, mABC, tA, tB, tC);
}
