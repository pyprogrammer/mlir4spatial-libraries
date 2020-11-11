package mlir_libraries.types

import argon.lang
import spatial.dsl

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

abstract class PureReadable[T] extends ReadableND[T] {
  def execute(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): T

  override def getInterface: Interface[T] = {
    new Interface[T] {
      override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): lang.Void = {new lang.Void}

      override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = execute(index, ens)
    }
  }
}

abstract class ReindexingReadable[T: spatial.dsl.Num](implicit state: argon.State) extends ReadableND[T] {
  def remapIndex(index: Seq[spatial.dsl.I32], ens:Set[spatial.dsl.Bit]): Seq[spatial.dsl.I32]
  def subReadable: ReadableND[T]

  override def getInterface: Interface[T] = {
    val subInterface = subReadable.getInterface
    new Interface[T] {
      override def enq(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): lang.Void = subInterface.enq(remapIndex(index, ens), ens)

      override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
        val v = subInterface.deq(remapIndex(index, ens), ens)
        mlir_libraries.debug_utils.TagVector("ReindexingOutput", Seq(v), ens)
        v
      }
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
