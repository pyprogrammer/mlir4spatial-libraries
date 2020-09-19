package mlir_libraries

import argon.stage
import forge.tags.stateful
import spatial.dsl
import spatial.node.SRAMRead

object types {
  trait Shaped {
    val shape: Seq[spatial.dsl.I32]
    def size(implicit state: argon.State): spatial.dsl.I32 = shape reduce {_ * _}
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

  @stateful implicit def NDCast[T: Bits, U: Bits](src: ReadableND[T])(implicit conv: argon.Cast[T, U]): ReadableND[U] = {
    new ReadableND[U] {
      override def apply(index: Seq[dsl.I32], ens: Set[dsl.Bit]): () => U = {
        val t = src(index, ens)
        () => t().to[U]
      }
      override lazy val shape = src.shape
    }
  }

  def ArrayToTensor[T](array: Array[T]): Tensor[T] =
    Tensor(values = array, shape = Array(array.length))
}
