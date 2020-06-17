package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._

trait Lattice {

  def Lattice[T: Num](lattice_kernel: scala.Array[scala.Array[scala.Double]], tp: String, shape: scala.Array[scala.Int],
                       units: Int, num_loop_dimensions: Int = 0
                     )(arg: ReadableND[T])(implicit state: argon.State): Readable2D[T] = {

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val dimensions = shape.length
    val parallel_dimensions = dimensions - num_loop_dimensions
    val strides = ComputeStrides(shape)
    val parallel_strides = strides.drop(num_loop_dimensions)

    // Ignore tp for now, always "hypercube"

    // The outer list is in order, the inner dimension of the array corresponds to an output vector unit.

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }.toSeq
    val params = LUT[OutputType](lattice_kernel.length, units)(param_list: _*)

    // needed to pass shape into readable def
    val lattice_shape = shape

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis = 1)(arg)
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

        val base_vec: Seq[ParameterIndex] = scala.Array.tabulate(num_loop_dimensions) { x =>
          (lattice_shape(x), HypercubeLattice.PO2Opt) match {
            case (2, true) =>
              0.to[ParameterIndex]
            case (shape, _) =>
              min(expanded_arg(batch, I32(x), unit).to[ParameterIndex], I32(shape - 1))
          }
        }

        // compute the base index
        // if we're not looping at all then ignore this.
        val base_index = if (num_loop_dimensions > 0) {
          (base_vec zip strides) map { case (b, s) => b * s.to[ParameterIndex] } reduceTree {
            _ + _
          }
        } else {
          0.to[ParameterIndex]
        }

        val parallel_residual_pairs = residualPairs.drop(num_loop_dimensions)

        val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(parallel_residual_pairs: _*)(_ * _)

        def recursive_fill(current_index: scala.Seq[ParameterIndex], base: ParameterIndex): OutputType = {
          val current_dimension = current_index.size
          if (current_dimension == num_loop_dimensions) {
            // finish directly

            // Get flat index for each (corner + origin)
            val indices: Seq[ParameterIndex] = corners map {
              corner =>
                val offset = ((corner zip parallel_strides) map {
                  case (cc, stride) =>
                    cc * stride
                }).sum.to[ParameterIndex]
                base_index + base + offset
            }

            // Get weighted sum
            hypervolumes.map(_.to[OutputType]).zip(indices).map {
              case (hv, i) =>
                hv * params(i, unit)
            }.reduceTree {
              _ + _
            }
          } else {
            val residual_pair = residualPairs(current_dimension)
            // finish recursively.
            Reduce(Reg[OutputType](0))(2 by 1) {
              bit =>
                val beqz = bit.infix_==(I32(0))
                val step = mux(beqz, 0.to[ParameterIndex], strides(current_dimension).to[ParameterIndex])
                // if bk == 0 then we take the weight to be 1-xk otherwise we use xk.
                val weight = mux(beqz, residual_pair(1), residual_pair(0))
                val recursive = recursive_fill(current_index :+ bit, base + step)

                recursive * weight.to[OutputType]
            } {
              _ + _
            }
          }
        }

        recursive_fill(scala.Seq.empty[ParameterIndex], argon.uconst[ParameterIndex](0))
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
    case h :: tail if tail.nonEmpty => (0 to h).flatMap { i => allCorners(tail, partials.map(_ ++ Seq(i))) }
    case h :: tail if tail.isEmpty => (0 to h).flatMap { i => partials.map(_ ++ Seq(i)) }
  }

  object CombinationTree {

    def apply[T](xs: Seq[T]*)(func: (T, T) => T): Seq[T] = combine(xs, func).head

    private def join[T](x1: Seq[T], x2: Seq[T], func: (T, T) => T): Seq[T] = x1.flatMap { a => x2.map { b => func(a, b) } }

    private def combine[T](xs: Seq[Seq[T]], func: (T, T) => T): Seq[Seq[T]] = xs.length match {
      case 0 => throw new Exception("Empty reduction level")
      case 1 => xs
      case len if len % 2 == 0 => combine(List.tabulate(len / 2) { i => join(xs(2 * i), xs(2 * i + 1), func) }, func)
      case len => combine(List.tabulate(len / 2) { i => join(xs(2 * i), xs(2 * i + 1), func) } :+ xs.last, func)
    }
  }

}
