package mlir_libraries

trait TensorLike[T] {
  val rank: Int
  val shape: Array[Int]
  def apply(index: Int*): T

  def to2DSeq: Seq[Seq[T]] = {
    assert(rank == 2, "to2DSeq only works on 2D Tensors")
    Range(0, shape.head) map {
      x_ind =>
        Range(0, shape(1)) map {
          y_ind => this(x_ind, y_ind)
        }
    }
  }

  def transpose(swapdims: (Int, Int) = (0, 1)): TensorLike[T] = {
    val that = this

    def map_index(ind: Int): Int = {
      ind match {
        case swapdims._1 => swapdims._2
        case swapdims._2 => swapdims._1
        case _ => ind
      }
    }

    new TensorLike[T] {
      val rank = that.rank
      val shape = (Range(0, that.rank) map {
        x => that.shape(map_index(x))
      }).toArray

      override def apply(index: Int*): T = {
        val corrected_index = Range(0, that.rank) map {
          dim => index(map_index(dim))
        }
        that(corrected_index:_*)
      }
    }
  }
}

case class Tensor[T](shape: Array[Int], values: Array[T]) extends TensorLike[T] {
  lazy val rank: Int = shape.length
  lazy val strides: Seq[Int] = utils.ComputeStrides(shape.reverse)
  def apply(index: Int*): T = {
    val flat_index = ((index zip strides) map { case (a, b) => a * b}).sum
    values(flat_index)
  }
}
