#include "cute/tensor.hpp"

#include "task.h"
#include "utils.h"

using namespace cute;

using TileShapeM = _64;
using TileShapeN = _32;
using TileShapeK = _32;

using TileShape = Shape<TileShapeM, TileShapeN, TileShapeK>;

using ElementAccumulator = float;

using TiledMma = TiledMMA<
    MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
    Layout<Shape<_16, _16, _1>>>;

// row-major A
using SmemLayoutAtomAW0 = Layout<Shape<TileShapeM, TileShapeK>, Stride<_1, Int<TileShapeM {} + 4>>>; // Padded

using SmemLayoutAW0 = decltype(tile_to_shape(
    SmemLayoutAtomAW0 {}, make_shape(TileShapeM {}, TileShapeK {})));

using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;

using GmemTiledCopyAW0 = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<ElementA>, ElementA> {},
    Layout<Shape<_32, _8>, Stride<_8, _1>> {},
    Layout<Shape<_1, _1>> {}));

// row-major AW1
using SmemLayoutAtomAW1 = Layout<Shape<TileShapeM, TileShapeN>, Stride<_1, Int<TileShapeM {} + 4>>>; // Padded

using SmemLayoutAW1 = decltype(tile_to_shape(
    SmemLayoutAtomAW1 {}, make_shape(TileShapeM {}, TileShapeN {})));

using SmemCopyAtomAW1 = Copy_Atom<DefaultCopy, ElementA>;

using GmemTiledCopyAW1 = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<ElementA>, ElementA> {},
    Layout<Shape<_32, _8>, Stride<_8, _1>> {},
    Layout<Shape<_1, _1>> {}));

// row-major B
using SmemLayoutAtomW1 = Layout<Shape<TileShapeN, TileShapeK>, Stride<_1, TileShapeN>>;

using SmemLayoutW1 = decltype(tile_to_shape(
    SmemLayoutAtomW1 {}, make_shape(TileShapeN {}, TileShapeK {})));

using SmemCopyAtomW1 = Copy_Atom<DefaultCopy, ElementB>;

using GmemTiledCopyW1 = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<ElementB>, ElementB> {},
    Layout<Shape<_32, _8>, Stride<_1, _32>> {},
    Layout<Shape<_1, _1>> {}));

// row-major C
using SmemLayoutAtomW2 = Layout<Shape<TileShapeN, TileShapeK>, Stride<_1, TileShapeN>>;

using SmemLayoutW2 = decltype(tile_to_shape(
    SmemLayoutAtomW2 {}, make_shape(TileShapeN {}, TileShapeK {})));

using SmemCopyAtomW2 = Copy_Atom<DefaultCopy, ElementC>;

using GmemTiledCopyW2 = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<ElementC>, ElementC> {},
    Layout<Shape<_32, _8>, Stride<_1, _32>> {},
    Layout<Shape<_1, _1>> {}));

// shared storage
struct SharedStorage {
    // features
    union {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutAW0>> smem_AW0;
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutAW1>> smem_AW1;
    };

    // weights
    union {
        cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutW1>> smem_W1;
        cute::array_aligned<ElementC, cute::cosize_v<SmemLayoutW2>> smem_W2;
    };
};

using TransformA = cute::identity;
using TransformB = cute::identity;

template <typename MAW0, typename MW1, typename MW2, typename MAW1, typename MAW1W2>
__global__ void fusedDeviceKernel(
    MAW0 mAW0_mk0, MW1 mW1_k1k0, MW2 mW2_k2k1, MAW1 mAW1_mk1, MAW1W2 mAW2_mk2)
{
    using X = Underscore;

    extern __shared__ char smemBuf[];

    int thread_idx = int(threadIdx.x);

    auto blk_shape = TileShape {}; // (BLK_M,BLK_N,BLK_K)
    auto [m_coord, n_coord, l_coord] = blockIdx;
    auto blk_coord_mnk = make_coord(m_coord, n_coord, _); // (m,n,k)

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smemBuf);

    Tensor gAW0 = local_tile(mAW0_mk0, blk_shape, blk_coord_mnk, Step<_1, X, _1> {}); // (BLK_M,BLK_K,k)

    Tensor sAW0 = make_tensor(make_smem_ptr(storage.smem_AW0.data()), SmemLayoutAW0 {}); // (BLK_M,BLK_K,PIPE)

    TiledMma tiledMMA;

    // A x W1

    Tensor gW1 = local_tile(mW1_k1k0, blk_shape, blk_coord_mnk, Step<X, _1, _1> {}); // (BLK_N,BLK_K,k)

    Tensor accumulatorsAW1 = partition_fragment_C(tiledMMA, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)
    clear(accumulatorsAW1);

    auto& src_accum_AW1 = accumulatorsAW1;

    auto k_tile_iter_AW1 = cute::make_coord_iterator(shape<2>(gW1));
    int k_tile_count_AW1 = size<2>(gW1);

    FIRST_THREAD_ONLY({
        println("k_tile_iter_AW1: ", shape<2>(gW1));
        println("k_tile_count_AW1: ", k_tile_count_AW1);
    })

    Tensor sW1 = make_tensor(make_smem_ptr(storage.smem_W1.data()), SmemLayoutW1 {}); // (BLK_N,BLK_K,PIPE)

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyAW0 gmem_tiled_copy_AW0;
    GmemTiledCopyW1 gmem_tiled_copy_W1;
    auto copy_AW0_thr = gmem_tiled_copy_AW0.get_slice(thread_idx);
    auto copy_W1_thr = gmem_tiled_copy_W1.get_slice(thread_idx);

    Tensor tAW0gAW0 = copy_AW0_thr.partition_S(gAW0); // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAW0sAW0 = copy_AW0_thr.partition_D(sAW0); // (ACPY,ACPY_M,ACPY_K)

    Tensor tW1gW1 = copy_W1_thr.partition_S(gW1); // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tW1sW1 = copy_W1_thr.partition_D(sW1); // (BCPY,BCPY_N,BCPY_K)

    // Allocate the register tiles for double buffering -- same shape as partitioned data
    Tensor tAW0rAW0 = make_fragment_like(tAW0sAW0); // (ACPY,ACPY_M,ACPY_K)
    Tensor tW1rW1 = make_fragment_like(tW1sW1); // (BCPY,BCPY_N,BCPY_K)

    // Tile MMA compute thread partitions and allocate accumulatorsAW1
    // TiledMma tiledMMA;
    auto thrMMA = tiledMMA.get_thread_slice(thread_idx);
    Tensor tAW1rAW0 = thrMMA.partition_fragment_A(sAW0); // (MMA,MMA_M,MMA_K)
    Tensor tAW1rW1 = thrMMA.partition_fragment_B(sW1); // (MMA,MMA_M,MMA_K)

    //
    // Copy Atom retiling
    //

    auto thr_copy_AW0 = make_tiled_copy_A(SmemCopyAtomA {}, tiledMMA).get_thread_slice(thread_idx);
    Tensor tAW1sAW0 = thr_copy_AW0.partition_S(sAW0);
    Tensor tAW1rAW0_copy_view = thr_copy_AW0.retile_D(tAW1rAW0);
    CUTE_STATIC_ASSERT_V(size<1>(tAW1sAW0) == size<1>(tAW1rAW0_copy_view)); // M

    auto thr_copy_W1 = make_tiled_copy_B(SmemCopyAtomW1 {}, tiledMMA).get_thread_slice(thread_idx);
    Tensor tAW1sW1 = thr_copy_W1.partition_S(sW1);
    Tensor tAW1rW1_copy_view = thr_copy_W1.retile_D(tAW1rW1);
    CUTE_STATIC_ASSERT_V(size<1>(tAW1sW1) == size<1>(tAW1rW1_copy_view)); // N

    //
    // Prologue
    //

    // Copy gmem to rmem for the first k_tile
    copy(gmem_tiled_copy_AW0, tAW0gAW0(_, _, _, *k_tile_iter_AW1), tAW0rAW0);
    copy(gmem_tiled_copy_W1, tW1gW1(_, _, _, *k_tile_iter_AW1), tW1rW1);

    if (--k_tile_count_AW1 > 0)
        ++k_tile_iter_AW1;

    // Copy rmem to smem
    copy(tAW0rAW0, tAW0sAW0);
    copy(tW1rW1, tW1sW1);

    // Clear accumulatorsAW1
    __syncthreads();

    // Load A, B smem->rmem for k=0
    copy(tAW1sAW0(_, _, 0), tAW1rAW0_copy_view(_, _, 0));
    copy(tAW1sW1(_, _, 0), tAW1rW1_copy_view(_, _, 0));

    //
    // Mainloop
    //

    // Size of the k-tiles's outer product mode (k)
    auto K_BLOCK_MAX = size<2>(tAW1rAW0);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count_AW1 > -1) {
        // Pipeline the outer products with a static for loop
        for_each(make_int_sequence<K_BLOCK_MAX> {}, [&](auto k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                __syncthreads();

                // Copy rmem to smem
                copy(tAW0rAW0, tAW0sAW0);
                copy(tW1rW1, tW1sW1);
                __syncthreads();
            }

            // Load A, B smem->rmem for k+1
            int k_block_next = (k_block + Int<1> {}) % K_BLOCK_MAX; // static

            copy(tAW1sAW0(_, _, k_block_next), tAW1rAW0_copy_view(_, _, k_block_next));
            copy(tAW1sW1(_, _, k_block_next), tAW1rW1_copy_view(_, _, k_block_next));

            if (k_block == 0) {
                // Copy gmem to rmem
                copy(gmem_tiled_copy_AW0, tAW0gAW0(_, _, _, *k_tile_iter_AW1), tAW0rAW0);
                copy(gmem_tiled_copy_W1, tW1gW1(_, _, _, *k_tile_iter_AW1), tW1rW1);
                if (--k_tile_count_AW1 > 0)
                    ++k_tile_iter_AW1;
            }

            // transform before compute
            cute::transform(tAW1rAW0(_, _, k_block), TransformA {});
            cute::transform(tAW1rW1(_, _, k_block), TransformB {});

            // Thread-level register gemm for k
            // disambiguate gemm (shared with the namespace name)
            cute::gemm(tiledMMA, accumulatorsAW1, tAW1rAW0(_, _, k_block), tAW1rW1(_, _, k_block), src_accum_AW1);
        });
    }

    FIRST_THREAD_ONLY({
        println("gridDim: ", gridDim);
        println("blockDim: ", blockDim);

        println("tiledMMA: ", tiledMMA);
        println("size(tiledMMA): ", size(tiledMMA));
        println("accumulatorsAW1: ", accumulatorsAW1);

        println("K_BLOCK_MAX: ", K_BLOCK_MAX);

        println("sAW0: ", sAW0);
        println("sW1: ", sW1);

        println("gmem_tiled_copy_AW0: ", gmem_tiled_copy_AW0);
        println("gmem_tiled_copy_W1: ", gmem_tiled_copy_W1);

        println("copy_AW0_thr: ", copy_AW0_thr);
        println("copy_W1_thr: ", copy_W1_thr);

        println("tAW0gAW0: ", tAW0gAW0);
        println("tAW0sAW0: ", tAW0sAW0);
        println("tW1gW1: ", tW1gW1);
        println("tW1sW1: ", tW1sW1);

        println("tAW0rAW0: ", tAW0rAW0);
        println("tW1rW1: ", tW1rW1);

        println("thrMMA: ", thrMMA);
        println("tAW1rAW0: ", tAW1rAW0);
        println("tAW1rW1: ", tAW1rW1);

        println("thr_copy_AW0: ", thr_copy_AW0);
        println("tAW1sAW0: ", tAW1sAW0);
        println("tAW1rAW0_copy_view: ", tAW1rAW0_copy_view);

        println("thr_copy_W1: ", thr_copy_W1);
        println("tAW1sW1: ", tAW1sW1);
        println("tAW1rW1_copy_view: ", tAW1rW1_copy_view);
    });

    {
        // -- eppilogue--
        Tensor gAW1_mk1 = local_tile(mAW1_mk1, blk_shape, make_coord(_, _, _), Step<_1, _1, X> {}); // (BLK_M,BLK_N,m,n)
        Tensor gAW1 = gAW1_mk1(_, _, m_coord, n_coord); // (BLK_M,BLK_N)

        // Partition source and destination tiles to match the accumulator partitioning
        // auto thrMMA = tiled_mma.get_thread_slice(thread_idx);
        Tensor tAW1gAW1 = thrMMA.partition_C(gAW1); // (VEC,THR_M,THR_N)

        // copy(accumulatorsAW1, tAW1gAW1);
        {
            // Make an identity coordinate tensor for predicating our output MN tile
            auto cAW1 = make_identity_tensor(make_shape(unwrap(shape<0>(gAW1)), unwrap(shape<1>(gAW1))));

            Tensor tAW1cAW1 = thrMMA.partition_C(cAW1);

            // Compute tile residues for predication
            auto m_max_coord = size<0>(mAW0_mk0) - size<0>(gAW0) * get<0>(blk_coord_mnk); // M - BLK_M * m_coord
            auto n_max_coord = size<0>(mW1_k1k0) - size<0>(gW1) * get<1>(blk_coord_mnk); // N - BLK_N * n_coord
            auto k_residue = size<1>(mAW0_mk0) - size<1>(gAW0) * size<2>(gAW0); // K - BLK_K * k_coord_max
            auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(accumulatorsAW1); ++i) {
                // FIRST_THREAD_ONLY({
                //     println("i: ", i, " tAW1cAW1(i): ", tAW1cAW1(i));
                // });

                if (elem_less(tAW1cAW1(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
                    tAW1gAW1(i) = accumulatorsAW1(i);
                }
            }
        }

        FIRST_THREAD_ONLY({
            println("gAW1_mk1: ", gAW1_mk1);
            println("gAW1: ", gAW1);
            println("tAW1gAW1: ", tAW1gAW1);
            // println("cAW1: ", cAW1);
            // println("tAW1cAW1: ", tAW1cAW1);
        })
    }

    __syncthreads();

    // AW1 x W2
    {
        Tensor gAW1 = local_tile(mAW1_mk1, blk_shape, blk_coord_mnk, Step<_1, X, _1> {}); // (BLK_M,BLK_K,k)

        Tensor sAW1 = make_tensor(make_smem_ptr(storage.smem_AW1.data()), SmemLayoutAW1 {}); // (BLK_M,BLK_K,PIPE)

        // A x W1

        Tensor gW2 = local_tile(mW2_k2k1, blk_shape, blk_coord_mnk, Step<X, _1, _1> {}); // (BLK_N,BLK_K,k)

        Tensor accumulatorsAW2 = partition_fragment_C(tiledMMA, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)
        clear(accumulatorsAW2);

        auto& src_accum_AW2 = accumulatorsAW2;

        auto k_tile_iter_AW2 = cute::make_coord_iterator(shape<2>(gW2));
        int k_tile_count_AW2 = size<2>(gW2);

        FIRST_THREAD_ONLY({
            println("k_tile_iter_AW2: ", shape<2>(gW2));
            println("k_tile_count_AW2: ", k_tile_count_AW2);
        })

        Tensor sW2 = make_tensor(make_smem_ptr(storage.smem_W2.data()), SmemLayoutW2 {}); // (BLK_N,BLK_K,PIPE)

        // Partition the copying of A and B tiles across the threads
        GmemTiledCopyAW1 gmem_tiled_copy_AW1;
        GmemTiledCopyW2 gmem_tiled_copy_W2;
        auto copy_AW1_thr = gmem_tiled_copy_AW1.get_slice(thread_idx);
        auto copy_W2_thr = gmem_tiled_copy_W2.get_slice(thread_idx);

        Tensor tAW1gAW1 = copy_AW1_thr.partition_S(gAW1); // (ACPY,ACPY_M,ACPY_K,k)
        Tensor tAW1sAW1 = copy_AW1_thr.partition_D(sAW1); // (ACPY,ACPY_M,ACPY_K)

        Tensor tW2gW2 = copy_W2_thr.partition_S(gW2); // (BCPY,BCPY_N,BCPY_K,k)
        Tensor tW2sW2 = copy_W2_thr.partition_D(sW2); // (BCPY,BCPY_N,BCPY_K)

        // Allocate the register tiles for double buffering -- same shape as partitioned data
        Tensor tAW1rAW1 = make_fragment_like(tAW1sAW1); // (ACPY,ACPY_M,ACPY_K)
        Tensor tW2rW2 = make_fragment_like(tW2sW2); // (BCPY,BCPY_N,BCPY_K)

        // Tile MMA compute thread partitions and allocate accumulatorsAW2
        // TiledMma tiledMMA;
        auto thrMMA = tiledMMA.get_thread_slice(thread_idx);
        Tensor tAW2rAW1 = thrMMA.partition_fragment_A(sAW1); // (MMA,MMA_M,MMA_K)
        Tensor tAW2rW2 = thrMMA.partition_fragment_B(sW2); // (MMA,MMA_M,MMA_K)

        //
        // Copy Atom retiling
        //

        auto thr_copy_AW1 = make_tiled_copy_A(SmemCopyAtomA {}, tiledMMA).get_thread_slice(thread_idx);
        Tensor tAW2sAW1 = thr_copy_AW1.partition_S(sAW1);
        Tensor tAW2rAW1_copy_view = thr_copy_AW1.retile_D(tAW2rAW1);
        CUTE_STATIC_ASSERT_V(size<1>(tAW2sAW1) == size<1>(tAW2rAW1_copy_view)); // M

        auto thr_copy_W2 = make_tiled_copy_B(SmemCopyAtomW1 {}, tiledMMA).get_thread_slice(thread_idx);
        Tensor tAW2sW2 = thr_copy_W2.partition_S(sW2);
        Tensor tAW2rW2_copy_view = thr_copy_W2.retile_D(tAW2rW2);
        CUTE_STATIC_ASSERT_V(size<1>(tAW2sW2) == size<1>(tAW2rW2_copy_view)); // N

        //
        // Prologue
        //

        // Copy gmem to rmem for the first k_tile
        copy(gmem_tiled_copy_AW1, tAW1gAW1(_, _, _, *k_tile_iter_AW2), tAW1rAW1);
        copy(gmem_tiled_copy_W2, tW2gW2(_, _, _, *k_tile_iter_AW2), tW2rW2);

        if (--k_tile_count_AW2 > 0)
            ++k_tile_iter_AW2;

        // Copy rmem to smem
        copy(tAW1rAW1, tAW1sAW1);
        copy(tW2rW2, tW2sW2);

        // Clear accumulatorsAW2
        __syncthreads();

        // Load A, B smem->rmem for k=0
        copy(tAW2sAW1(_, _, 0), tAW2rAW1_copy_view(_, _, 0));
        copy(tAW2sW2(_, _, 0), tAW2rW2_copy_view(_, _, 0));

        //
        // Mainloop
        //

        // Size of the k-tiles's outer product mode (k)
        auto K_BLOCK_MAX = size<2>(tAW2rAW1);

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count_AW2 > -1) {
            // Pipeline the outer products with a static for loop
            for_each(make_int_sequence<K_BLOCK_MAX> {}, [&](auto k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    __syncthreads();

                    // Copy rmem to smem
                    copy(tAW1rAW1, tAW1sAW1);
                    copy(tW2rW2, tW2sW2);
                    __syncthreads();
                }

                // Load A, B smem->rmem for k+1
                int k_block_next = (k_block + Int<1> {}) % K_BLOCK_MAX; // static

                copy(tAW2sAW1(_, _, k_block_next), tAW2rAW1_copy_view(_, _, k_block_next));
                copy(tAW2sW2(_, _, k_block_next), tAW2rW2_copy_view(_, _, k_block_next));

                if (k_block == 0) {
                    // Copy gmem to rmem
                    copy(gmem_tiled_copy_AW1, tAW1gAW1(_, _, _, *k_tile_iter_AW2), tAW1rAW1);
                    copy(gmem_tiled_copy_W2, tW2gW2(_, _, _, *k_tile_iter_AW2), tW2rW2);
                    if (--k_tile_count_AW2 > 0)
                        ++k_tile_iter_AW2;
                }

                // transform before compute
                cute::transform(tAW2rAW1(_, _, k_block), TransformA {});
                cute::transform(tAW2rW2(_, _, k_block), TransformB {});

                // Thread-level register gemm for k
                // disambiguate gemm (shared with the namespace name)
                cute::gemm(tiledMMA, accumulatorsAW2, tAW2rAW1(_, _, k_block), tAW2rW2(_, _, k_block), src_accum_AW2);
            });
        }

        FIRST_THREAD_ONLY({
            println("accumulatorsAW2: ", accumulatorsAW2);

            println("K_BLOCK_MAX: ", K_BLOCK_MAX);

            println("sAW1: ", sAW1);
            println("sW2: ", sW2);

            println("gmem_tiled_copy_AW1: ", gmem_tiled_copy_AW1);
            println("gmem_tiled_copy_W2: ", gmem_tiled_copy_W2);

            // println("copy_AW1_thr: ", copy_AW1_thr);
            // println("copy_W2_thr: ", copy_W2_thr);

            // println("tAW1gAW1: ", tAW1gAW1);
            // println("tAW1sAW1: ", tAW1sAW1);
            // println("tW2gW2: ", tW2gW2);
            // println("tW2sW2: ", tW2sW2);

            // println("tAW1rAW1: ", tAW1rAW1);
            // println("tW2rW2: ", tW2rW2);

            // println("thrMMA: ", thrMMA);
            // println("tAW2rAW1: ", tAW2rAW1);
            // println("tAW2rW2: ", tAW2rW2);

            // println("thr_copy_AW1: ", thr_copy_AW1);
            // println("tAW2sAW1: ", tAW2sAW1);
            // println("tAW2rAW1_copy_view: ", tAW2rAW1_copy_view);

            // println("thr_copy_W2: ", thr_copy_W2);
            // println("tAW2sW2: ", tAW2sW2);
            // println("tAW2rW2_copy_view: ", tAW2rW2_copy_view);
        });

        {
            // -- eppilogue--
            Tensor gAW2_mk2 = local_tile(mAW2_mk2, blk_shape, make_coord(_, _, _), Step<_1, _1, X> {}); // (BLK_M,BLK_N,m,n)
            Tensor gAW2 = gAW2_mk2(_, _, m_coord, n_coord); // (BLK_M,BLK_N)

            // Partition source and destination tiles to match the accumulator partitioning
            // auto thrMMA = tiled_mma.get_thread_slice(thread_idx);
            Tensor tAW2gAW2 = thrMMA.partition_C(gAW2); // (VEC,THR_M,THR_N)

            // copy(accumulatorsAW2, tAW2gAW2);
            {
                // Make an identity coordinate tensor for predicating our output MN tile
                auto cAW2 = make_identity_tensor(make_shape(unwrap(shape<0>(gAW2)), unwrap(shape<1>(gAW2))));

                Tensor tAW2cAW2 = thrMMA.partition_C(cAW2);

                // Compute tile residues for predication
                auto m_max_coord = size<0>(mAW0_mk0) - size<0>(gAW0) * get<0>(blk_coord_mnk); // M - BLK_M * m_coord
                auto n_max_coord = size<0>(mW2_k2k1) - size<0>(gW2) * get<1>(blk_coord_mnk); // N - BLK_N * n_coord
                // auto k_residue = size<1>(mAW2_mk2) - size<1>(gAW2) * size<2>(gAW2); // K - BLK_K * k_coord_max
                // auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);
                auto residue_mn = make_tuple(m_max_coord, n_max_coord);

                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(accumulatorsAW2); ++i) {
                    // FIRST_THREAD_ONLY({
                    //     println("i: ", i, " tAW2cAW2(i): ", tAW2cAW2(i));
                    // });

                    if (elem_less(tAW2cAW2(i), residue_mn)) {
                        tAW2gAW2(i) = accumulatorsAW2(i);
                    }
                }
            }

            FIRST_THREAD_ONLY({
                println("gAW2_mk2: ", gAW2_mk2);
                println("gAW2: ", gAW2);
                println("tAW2gAW2: ", tAW2gAW2);
                // println("cAW2: ", cAW2);
                // println("tAW1cAW1: ", tAW1cAW1);
            })
        }
    }
}

void fusedKernel(cudaStream_t stream, const Task& task)
{
    auto mA = make_tensor(make_gmem_ptr(task.devA), make_shape(task.m, task.k0), GenRowMajor {});

    auto mB = make_tensor(make_gmem_ptr(task.devB), make_shape(task.k1, task.k0), GenColMajor {});

    auto mC = make_tensor(make_gmem_ptr(task.devC), make_shape(task.n, task.k1), GenColMajor {});

    auto mAB = make_tensor(make_gmem_ptr(task.devAB), make_shape(task.m, task.k1), GenRowMajor {});

    auto mABC = make_tensor(make_gmem_ptr(task.devABC), make_shape(task.m, task.n), GenRowMajor {});

    auto problem_shape = make_shape(task.m, task.k0, task.k1, task.n);

    dim3 const grid = dim3(
        cute::size(cute::ceil_div(task.m, cute::shape<0>(TileShape {}))),
        cute::size(cute::ceil_div(std::max(task.k1, task.n), cute::shape<1>(TileShape {}))),
        1);

    dim3 const block = dim3(int(cute::size(TiledMma {})), 1, 1);

    int smem_size = sizeof(SharedStorage);

    println("smem_size: ", smem_size);

    fusedDeviceKernel<<<grid, block, smem_size, stream>>>(
        mA, mB, mC, mAB, mABC);
}
