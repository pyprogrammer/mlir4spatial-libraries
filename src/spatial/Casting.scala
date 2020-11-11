package mlir_libraries
import types._
import spatial.dsl._
import forge.tags.stateful

trait Casting {
  @stateful def Cast[From: Num, To: Num](arg: ReadableND[From]): ReadableND[To] = {
    new ReadableND[To] {
      override def getInterface: Interface[To] = {
        val inter = arg.getInterface
        new Interface[To] {
          override def enq(index: Seq[I32], ens: Set[Bit]): Void = inter.enq(index, ens)

          override def deq(index: Seq[I32], ens: Set[Bit]): To = inter.deq(index, ens).to[To]
        }
      }
      override lazy val shape = arg.shape
    }
  }
}
