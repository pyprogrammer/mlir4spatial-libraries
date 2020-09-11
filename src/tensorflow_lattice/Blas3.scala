package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._

import mlir_libraries.{Tensor => MLTensor}

trait Blas3 {
  def Dense[T:Num](bias: MLTensor[Double], kernel: MLTensor[Double])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    val bias_array = bias.to1DSeq
    val kernel_array = kernel.to2DSeq
    val input_units = kernel_array.length
    val output_units = {
      val output_shapes = (kernel_array map {_.length}).distinct
      assert(output_shapes.length == 1, s"Found multiple possible output shapes: $output_shapes")
      output_shapes.head
    }

    val kernel_lut = LUT[T](input_units, output_units)((kernel_array.flatten map { x => Bits(x.toUnchecked[T])}):_*)
    val bias_LUT = LUT[T](bias_array.length)((bias_array map { x => Bits(x.toUnchecked[T])}):_*)

    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        val batch = index.head
        val dim = index.last
        // Goes from (batch x input units) x kernel^T (input x output) -> batch x output units.
        // Given C = AB, we have C_ij = Sum_k A_ik B_kj
        val reads = Range(0, input_units) map { i => arg(Seq(batch, I32(i)), ens)}

        {
          import spatial.dsl._
          print(r"Dense Layer:")
          reads map {read => print(r" ${read()}")}
          println("")
        }

        () =>
          {
            bias_LUT(dim) + (Range(0, input_units) map { i => reads(i)() * kernel_lut(I32(i), dim)}).reduceTree{_+_}
          }
      }
      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(output_units))
    }
  }
}
