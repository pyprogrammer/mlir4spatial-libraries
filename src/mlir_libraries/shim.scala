package mlir_libraries

import argon.stage
import forge.tags.stateful
import spatial.node.SRAMRead

object types {
  trait Shaped {
    val shape: Seq[spatial.dsl.I32]
  }

  trait ReadableND[T] extends Shaped {
    def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T
  }
}

object ConversionImplicits {
  import types._
  import spatial.libdsl._

  @stateful implicit def SRAMToND[T:Bits, C[U]](rm: SRAM[T,C])(implicit srcCtx: SrcCtx): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        assert(index.length == shape.length, f"Cannot read from a ReadableND converted to ND: ${index}, $srcCtx")
        () => stage(SRAMRead(rm, index, ens))
      }
      override lazy val shape: Seq[I32] = rm.dims
    }
  }

  def ArrayToTensor[T](array: Array[T]): Tensor[T] =
    Tensor(values = array, shape = Array(array.length))
}
