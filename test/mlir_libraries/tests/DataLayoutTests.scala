package mlir_libraries.tests

import mlir_libraries.{CoprocessorScope, Tensor => MLTensor}
import spatial.dsl._
import mlir_libraries.types.TypeImplicits._

object TransposeTest {
  val rank = 16
  val iterator = mlir_libraries.utils.CartesianProduct((0 until rank), (0 until rank), (0 until rank))

  // Swap index 0 and 2
  val testInput = iterator map {ind => ind(0) * rank * rank + ind(1) * rank + ind(2)}
}

@spatial class TransposeTest extends SpatialTest {

  type T = spatial.dsl.FixPt[TRUE, _2, _30]
  val rank = TransposeTest.rank

  def main(args: Array[String]): Unit = {
    val outDRAM = DRAM[I32](I32(rank), I32(rank), I32(rank))
    Accel {
      val data = RegFile(I32(rank), I32(rank), I32(rank), (TransposeTest.testInput map { x => I32(x) }))
      val out = SRAM[I32](I32(rank), I32(rank), I32(rank))
      val trans = tensorflow_lattice.tf.Transpose(MLTensor(shape=scala.Array(2), values=scala.Array(0, 2)))(data)
      val int = trans.getInterface
      Foreach(rank by 1, rank by 1, rank by 1) {
        (i, j, k) =>
          out(i, j, k) = int.deq(Seq(i, j, k), Set(Bit(true)))
      }
      outDRAM store out
    }
    val fetched = getTensor3(outDRAM)
    val golden = Tensor3.tabulate(I32(rank), I32(rank), I32(rank)) {
      (i, j, k) => k + rank * j + rank * rank * i
    }
    Foreach(rank by 1, rank by 1, rank by 1) {
      (i, j, k) =>
        if (golden(i, j, k) != fetched(i, j, k)) {
          println(r"Mismatch at $i, $j, $k: Wanted: ${golden(i, j, k)}, Received: ${fetched(i, j, k)}")
        }
    }
    assert(Bit(true))
  }
}

@spatial class TransposeTest2 extends SpatialTest {

  type T = spatial.dsl.FixPt[TRUE, _2, _30]
  val rank = TransposeTest.rank

  def main(args: Array[String]): Unit = {
    val outDRAM = DRAM[I32](I32(rank), I32(rank), I32(rank))
    Accel {
      val data = RegFile(I32(rank), I32(rank), I32(rank), (TransposeTest.testInput map { x => I32(x) }))
      val out = SRAM[I32](I32(rank), I32(rank), I32(rank))
      val trans = tensorflow_lattice.tf.Transpose(MLTensor(shape=scala.Array(2), values=scala.Array(2, 0)))(data)
      val int = trans.getInterface
      Foreach(rank by 1, rank by 1, rank by 1) {
        (i, j, k) =>
          out(i, j, k) = int.deq(Seq(i, j, k), Set(Bit(true)))
      }
      outDRAM store out
    }
    val fetched = getTensor3(outDRAM)
    val golden = Tensor3.tabulate(I32(rank), I32(rank), I32(rank)) {
      (i, j, k) => k + rank * j + rank * rank * i
    }
    Foreach(rank by 1, rank by 1, rank by 1) {
      (i, j, k) =>
        if (golden(i, j, k) != fetched(i, j, k)) {
          println(r"Mismatch at $i, $j, $k: Wanted: ${golden(i, j, k)}, Received: ${fetched(i, j, k)}")
        }
    }
    assert(Bit(true))
  }
}

