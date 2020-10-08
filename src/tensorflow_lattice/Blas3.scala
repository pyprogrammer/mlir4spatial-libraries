package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._
import mlir_libraries.{CoprocessorScope, Tensor => MLTensor}

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

    val kernel_lut =LUT[T](input_units, output_units)((kernel_array.flatten map { x => Bits(x.toUnchecked[T])}):_*)
    val bias_LUT = LUT[T](bias_array.length)((bias_array map { x => Bits(x.toUnchecked[T])}):_*)

    new ReadableND[T] {

      override def getInterface: Interface[T] = {
        val interfaces = Range(0, input_units) map {i => arg.getInterface}

        new Interface[T] {
          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            val batch = index.head
            interfaces.zipWithIndex foreach {
              case(interface, i) => interface.enq(Seq(batch, I32(i)), ens)
            }

            new Void
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            val batch = index.head
            val dim = index.last

            val reads = interfaces.zipWithIndex map {
              case (interface, i) =>
                interface.deq(Seq(batch, I32(i)), ens) * kernel_lut(I32(i), dim)
            }

            bias_LUT(dim) + reads.reduceTree {_+_}
          }
        }
      }
      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(output_units))
    }
  }
}
