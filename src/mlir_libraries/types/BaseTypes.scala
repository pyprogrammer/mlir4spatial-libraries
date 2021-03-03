package mlir_libraries.types

import argon.lang
import spatial.dsl

trait Shaped {
  val shape: Seq[spatial.dsl.I32]
  def size(implicit state: argon.State): spatial.dsl.I32 = shape reduce {_ * _}
  @forge.tags.api def isValid(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]) = {
//    (shape zip index).zipWithIndex foreach {
//      case ((shape, ind), dim) =>
//        val matches = (ind >= spatial.dsl.I32(0)) && (ind < shape)
//        spatial.dsl.assertIf(ens, !matches, Some(spatial.dsl.Text(s"Out of Bounds at dimension $dim")))
//    }
  }
}

trait FunctionLike[InT, OutT] {
  def enq(input: InT, ens: Set[spatial.dsl.Bit]): argon.lang.Void
  def deq(input: InT, ens: Set[spatial.dsl.Bit]): OutT
}

trait Interface[T] extends FunctionLike[Seq[spatial.dsl.I32], T]

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
