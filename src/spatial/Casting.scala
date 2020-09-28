package mlir_libraries
import mlir_libraries.types.ReadableND
import spatial.dsl._
import forge.tags.stateful

trait Casting {
  @stateful def Cast[From: Num, To: Num](arg: ReadableND[From]): ReadableND[To] = {
    new ReadableND[To] {
      override def apply(index: Seq[I32], ens: Set[Bit]): () => To = {
        val tmp = arg(index, ens)
        () => tmp().to[To]
      }
      override val shape: Seq[I32] = arg.shape
    }
  }
}
