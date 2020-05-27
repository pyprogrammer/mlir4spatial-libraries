package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._

import scala.reflect.ClassTag

trait Blas3 {
  def Dense[T:Num:ClassTag](bias: Array[Double], kernel: Array[Array[Double]])(arg: Readable2D[T])(implicit state: argon.State): Readable2D[T] = {
    val input_units = kernel.length
    val output_units = {
      val output_shapes = (kernel map {_.length}).distinct
      assert(output_shapes.length == 1, s"Found multiple possible output shapes: $output_shapes")
      output_shapes.head
    }

    val kernel_lut = LUT[T](input_units, output_units)((kernel.flatten map { x => Bits(x.toUnchecked[T])}):_*)
    val bias_LUT = LUT[T](bias.length)((bias map {x => Bits(x.toUnchecked[T])}):_*)

    new Readable2D[T] {
      override def apply(batch: I32, dim: I32): T = {
        // Goes from (batch x input units) x kernel^T (input x output) -> batch x output units.
        // Given C = AB, we have C_ij = Sum_k A_ik B_kj
        println("Executing Dense")
        bias_LUT(dim) + (Array.tabulate(input_units) { i => arg(batch, i) * kernel_lut(I32(i), dim)}).reduce{_+_}
      }
      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(output_units))
    }
  }
}
