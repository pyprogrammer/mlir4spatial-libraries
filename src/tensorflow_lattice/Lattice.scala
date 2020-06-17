package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._

trait Lattice {

  protected def ComputeStrides(dimensions: IndexedSeq[Int]): IndexedSeq[Int] = {
    val strides: scala.Array[Int] = scala.Array.fill(dimensions.length){1}
    scala.Range(1, dimensions.length, 1) foreach {
      d => {
        strides(d) = strides(d-1) * dimensions(d-1)
      }
    }
    strides.reverse
  }

  def Lattice[T : Num](lattice_kernel: scala.Array[scala.Array[scala.Double]], tp: String, shape: scala.Array[scala.Int], units: Int)(arg:ReadableND[T])(implicit state:argon.State): Readable2D[T] = {

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val dimensions = shape.length
    val parallel_dimensions = dimensions - num_loop_dimensions
    val strides = ComputeStrides(shape)

    // Ignore tp for now, always "hypercube"

    // The outer list is in order, the inner dimension of the array corresponds to an output vector unit.

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T])}.toSeq
    val params = LUT[OutputType](lattice_kernel.length, units)(param_list:_*)

    // needed to pass shape into readable def
    val lattice_shape = shape

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis=1)(arg)
    }
    assert(expanded_arg.shape.length == 3)

    // Get all vertices of hypercube and reverse so that these are opposite the hypervolumes
    val corners: Seq[Seq[scala.Int]] = HypercubeLattice.allCorners(Seq.fill(parallel_dimensions)(1)).reverse

    new Readable2D[T] {
      override def apply(batch: I32, unit: I32): T = {

        val residualPairs = Seq.tabulate(dimensions) { i =>
          val x = expanded_arg(batch, unit, i).to[ResidualType]
          val floored = floor(x).to[AccumResidualType]
          val diff = x - floored
          scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
        }
        // Compute all hypervolumes in binary counting order (000, 001, 010, 011, etc.)
        val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(residualPairs: _*)(_ * _)
        // Compute hypercube origin
        // if the dimension is 0, then we optimize by setting the base index to 0 instead.
        val base: Seq[ParameterIndex] = scala.Array.tabulate(dimensions) { x =>
          (lattice_shape(x), HypercubeLattice.PO2Opt) match {
            case (2, true) =>
              0.to[ParameterIndex]
            case _ =>
              expanded_arg(batch, x, unit).to[ParameterIndex]
          }
        }

        // Get flat index for each (corner + origin)
        val indices: Seq[ParameterIndex] = corners map { c =>
          val corner = (base zip c.map(_.to[ParameterIndex])) map { case (a, b) => a + b }
          (corner zip strides) map { case (cc, stride) =>
            cc * stride
          } reduce {
            _ + _
          }
        }

        // Get weighted sum
        hypervolumes.map(_.to[OutputType]).zip(indices).map {
          case (hv, i) =>
            hv * params(i, unit)
        }.reduceTree {
          _ + _
        }
      }

      // A lattice goes from (batch, dim, unit) -> (batch, unit)
      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(units))
    }
  }

  protected def ComputeStrides(dimensions: IndexedSeq[Int]): IndexedSeq[Int] = {
    val strides: scala.Array[Int] = scala.Array.fill(dimensions.length) {
      1
    }
    scala.Range(1, dimensions.length, 1) foreach {
      d => {
        strides(d) = strides(d - 1) * dimensions(d - 1)
      }
    }
    strides.reverse
  }
}

private object HypercubeLattice {
  val PO2Opt = sys.env.getOrElse("PO2Opt", "true") == "true"
  def allCorners(maxes: Seq[scala.Int], partials: Seq[Seq[scala.Int]] = Seq(Seq.empty)): Seq[Seq[scala.Int]] = maxes match {
    case Nil => Nil
    case h::tail if tail.nonEmpty => (0 to h).flatMap{i => allCorners(tail, partials.map(_ ++ Seq(i)))}
    case h::tail if tail.isEmpty => (0 to h).flatMap{i => partials.map(_ ++ Seq(i))}
  }

  object CombinationTree {

    private def join[T](x1: Seq[T], x2: Seq[T], func: (T,T) => T): Seq[T] = x1.flatMap{a => x2.map{b => func(a,b)}}

    private def combine[T](xs: Seq[Seq[T]], func: (T,T) => T): Seq[Seq[T]] = xs.length match {
      case 0 => throw new Exception("Empty reduction level")
      case 1 => xs
      case len if len % 2 == 0 => combine(List.tabulate(len/2){i => join(xs(2*i), xs(2*i+1), func)}, func)
      case len => combine(List.tabulate(len/2){i => join(xs(2*i), xs(2*i+1), func) } :+ xs.last, func)
    }

    def apply[T](xs: Seq[T]*)(func: (T,T) => T): Seq[T] = combine(xs, func).head
  }
}
