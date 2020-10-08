package mlir_libraries

import argon.{lang, stage}
import forge.tags.stateful
import spatial.dsl
import spatial.node.SRAMRead

object types {
  trait Shaped {
    val shape: Seq[spatial.dsl.I32]
    def size(implicit state: argon.State): spatial.dsl.I32 = shape reduce {_ * _}
  }

  trait Interface[T] {

    // Deq must be in same order as Enq. Deq having this interface is simply for 0-cost abstractions.
    def enq(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): argon.lang.Void
    def deq(index: Seq[spatial.dsl.I32], ens:Set[spatial.dsl.Bit]): T
  }

  trait ReadableND[T] extends Shaped {
    def getInterface: Interface[T]
  }

  trait PureReadable[T] extends ReadableND[T] {
    def execute(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): T

    override def getInterface: Interface[T] = {
      new Interface[T] {
        override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): lang.Void = {new lang.Void}

        override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = execute(index, ens)
      }
    }
  }

  trait ReindexingReadable[T] extends ReadableND[T] {
    def remapIndex(index: Seq[spatial.dsl.I32]): Seq[spatial.dsl.I32]
    def subReadable: ReadableND[T]

    override def getInterface: Interface[T] = {
      val subInterface = subReadable.getInterface
      new Interface[T] {
        override def enq(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): lang.Void = subInterface.enq(remapIndex(index), ens)

        override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = subInterface.deq(remapIndex(index), ens)
      }
    }
  }

  trait ElementWiseReadable[T] extends ReadableND[T] {
    def func(x: T): T
    def subReadable: ReadableND[T]

    override def getInterface: Interface[T] = {
      val subInterface = subReadable.getInterface
      new Interface[T] {
        override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): lang.Void = subInterface.enq(index, ens)

        override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = func(subInterface.deq(index, ens))
      }
    }

    override val shape: Seq[dsl.I32] = subReadable.shape
  }
}

object ConversionImplicits {
  import types._
  import spatial.libdsl._

  @stateful implicit def SRAMToND[T:Bits, C[U]](rm: SRAM[T,C])(implicit srcCtx: SrcCtx): ReadableND[T] = {
    new PureReadable[T] {
      override def execute(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]) = {
        assert(index.length == shape.length, f"Cannot read from a ReadableND converted to ND: ${index}, $srcCtx")
        stage(SRAMRead(rm, index, ens))
      }
      override lazy val shape: Seq[I32] = rm.dims
    }
  }

  def ArrayToTensor[T](array: Array[T]): Tensor[T] =
    Tensor(values = array, shape = Array(array.length))
}
