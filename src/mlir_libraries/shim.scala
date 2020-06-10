package mlir_libraries

import forge.tags.stateful

object types {
  trait Shaped {
    val shape: Seq[spatial.dsl.I32]
  }

  trait ReadableND[T] extends Shaped {
    def apply(index: spatial.dsl.I32*): T
  }

  trait Readable1D[T] extends Shaped {
    def apply(d0: spatial.dsl.I32): T
  }

  trait Readable2D[T] extends Shaped {
    def apply(d0: spatial.dsl.I32, d1: spatial.dsl.I32): T
  }
}

object ConversionImplicits {
  import types._
  import spatial.libdsl._
  @stateful implicit def RR1[T](rm : SRAM1[T]): Readable1D[T] = {
    new Readable1D[T] {
      override def apply(d0: I32): T = rm(d0)
      override lazy val shape: Seq[I32] = rm.dims
    }
  }
  @stateful implicit def RR2[T](rm : SRAM2[T]): Readable2D[T] = {
    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = rm(d0, d1)
      override lazy val shape: Seq[I32] = rm.dims
    }
  }

  @stateful implicit def R2N[T](rm: Readable2D[T]): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: spatial.dsl.I32*): T = {
        assert(index.length == shape.length, f"Cannot read from a Readable2D converted to ND: ${index}")
        rm(index(0), index(1))
      }
      override lazy val shape: Seq[I32] = rm.shape
    }
  }

  @stateful implicit def N2R[T](rm: ReadableND[T]): Readable2D[T] = {
    assert(rm.shape.length == 2, f"Cannot convert an ND to 2D readable. (N = ${rm.shape.length})")
    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = rm(d0, d1)
      override lazy val shape: Seq[I32] = rm.shape
    }
  }
}
