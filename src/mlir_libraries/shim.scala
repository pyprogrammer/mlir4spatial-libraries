package mlir_libraries

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
  implicit def RR1[T](rm : SRAM1[T])(implicit state: argon.State) : Readable1D[T] = {
    new Readable1D[T] {
      override def apply(d0: I32): T = rm(d0)
      override lazy val shape: Seq[I32] = rm.dims
    }
  }
  implicit def RR2[T](rm : SRAM2[T])(implicit state: argon.State) : Readable2D[T] = {
    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = rm(d0, d1)
      override lazy val shape: Seq[I32] = rm.dims
    }
  }

  implicit def R2N[T](rm: Readable2D[T])(implicit state: argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: spatial.dsl.I32*): T = {
        assert(index.length == shape.length)
        rm(index(0), index(1))
      }
      override lazy val shape: Seq[I32] = rm.shape
    }
  }

  implicit def N2R[T](rm: ReadableND[T])(implicit state: argon.State): Readable2D[T] = {
    assert(rm.shape.length == 2)
    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = rm(d0, d1)
      override lazy val shape: Seq[I32] = rm.shape
    }
  }
}
