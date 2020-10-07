package tensorflow_lattice

import mlir_libraries.types._
import mlir_libraries.utils.checkpoint
import spatial.libdsl._
import mlir_libraries.{Tensor => MLTensor}
import _root_.spatial.dsl

trait Lattice {

  def Lattice[T: Num](lattice_kernel: MLTensor[scala.Double], tp: String, shape: MLTensor[scala.Int],
                       units: Int)(arg: ReadableND[T])(implicit state: argon.State, config: mlir_libraries.OptimizationConfig): ReadableND[T] = {

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val dimensions = shape.shape.head
    val num_loop_dimensions = scala.math.min(config.lattice_loops, dimensions - 1)
    val parallel_dimensions = dimensions - num_loop_dimensions
    val strides = mlir_libraries.utils.ComputeStrides(shape.flatten.toIndexedSeq)
    val parallel_strides = strides.drop(num_loop_dimensions)

    // Ignore tp for now, always "hypercube"

    // The outer list is in order, the inner dimension of the array corresponds to an output vector unit.

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }
    val params = LUT[OutputType](lattice_kernel.shape.head, units)(param_list: _*)

    // needed to pass shape into readable def
    val lattice_shape = shape

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis = 1)(arg)
    }
    assert(expanded_arg.shape.length == 3, "Expanded arg should have rank 3")

    // Get all vertices of hypercube and reverse so that these are opposite the hypervolumes
    val corners: Seq[Seq[scala.Int]] = HypercubeLattice.allCorners(Seq.fill(parallel_dimensions)(1)).reverse

    new ReadableND[T] {
      override def getInterface: Interface[T] = {
        val parallelInterfaces = Range(0, parallel_dimensions) map {_ => arg.getInterface}
        val sequentialInterface = num_loop_dimensions match {
          case 0 => None
          case other => Some(arg.getInterface)
        }

        new Interface[T] {
          override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): Void = {
            val batch = index(index.length - 2)
            val unit = index.last
            Range(num_loop_dimensions, dimensions) zip parallelInterfaces foreach {
              case (dim, interface) => interface.enq(Seq(batch, unit, I32(dim)), ens)
            }
          }

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
              val batch = index(index.length - 2)
              val unit = index.last

              val new_enable = if (ens.isEmpty) Set(Bit(true)) else ens

              def recursive_fill(current_index: scala.Seq[ParameterIndex], base: ParameterIndex): OutputType = {
                val current_dimension = current_index.size
                checkpoint(s"LatticeLoop${current_dimension}_begin")
                val result: OutputType = if (current_dimension == num_loop_dimensions) {
                  // finish directly

                  val remainingInputs = Range(num_loop_dimensions, dimensions) map {
                    dim => expanded_arg(Seq(batch, unit, I32(dim)), new_enable)
                  } map {x => x()}

                  val parallelResidualPairs = remainingInputs map {
                    value =>
                      val floored = floor(value).to[AccumResidualType]
                      val diff = value - floored
                      scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
                  }

                  println(s"Parallel pairs: $parallelResidualPairs, length: ${parallelResidualPairs.length}")

                  val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(parallelResidualPairs: _*)(_ * _)

                  val base_vec = (Range(num_loop_dimensions, dimensions) zip remainingInputs) map {
                    case (dim, inpt) =>
                      (lattice_shape(dim), mlir_libraries.Options.PO2Opt) match {
                        case (2, true) =>
                          0.to[ParameterIndex]
                        case (shape, _) =>
                          min(inpt.to[ParameterIndex], I32(shape - 1))
                      }
                  }

                  println(s"Base Vec: $base_vec, Strides: ${strides.drop(num_loop_dimensions)}")

                  val base_index = (base_vec zip strides.drop(num_loop_dimensions)) map { case (b, s) => b * s.to[ParameterIndex] } reduceTree { _ + _ }

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
                  val input_staged = expanded_arg(Seq(batch, unit, I32(current_dimension)), new_enable)
                  // finish recursively.
                  Pipe.Reduce(Reg[OutputType](0))(2 by 1) {
                    bit =>
                      val input = input_staged()
                      val floored = floor(input).to[AccumResidualType]
                      val diff = input - floored
                      val residual_pair = scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
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

                checkpoint(s"LatticeLoop${current_dimension}_end")
                result
              }
              recursive_fill(scala.Seq.empty[ParameterIndex], argon.uconst[ParameterIndex](0))
            }
          }
        }
      }

//      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
//        val batch = index(index.length - 2)
//        val unit = index.last
//
//        val new_enable = if (ens.isEmpty) Set(Bit(true)) else ens
//
//        def recursive_fill(current_index: scala.Seq[ParameterIndex], base: ParameterIndex): OutputType = {
//          val current_dimension = current_index.size
//          checkpoint(s"LatticeLoop${current_dimension}_begin")
//          val result: OutputType = if (current_dimension == num_loop_dimensions) {
//          // finish directly
//
//          val remainingInputs = Range(num_loop_dimensions, dimensions) map {
//            dim => expanded_arg(Seq(batch, unit, I32(dim)), new_enable)
//          } map {x => x()}
//
//          val parallelResidualPairs = remainingInputs map {
//            value =>
//              val floored = floor(value).to[AccumResidualType]
//              val diff = value - floored
//              scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
//          }
//
//          println(s"Parallel pairs: $parallelResidualPairs, length: ${parallelResidualPairs.length}")
//
//          val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(parallelResidualPairs: _*)(_ * _)
//
//          val base_vec = (Range(num_loop_dimensions, dimensions) zip remainingInputs) map {
//            case (dim, inpt) =>
//              (lattice_shape(dim), mlir_libraries.Options.PO2Opt) match {
//                case (2, true) =>
//                  0.to[ParameterIndex]
//                case (shape, _) =>
//                  min(inpt.to[ParameterIndex], I32(shape - 1))
//              }
//          }
//
//          println(s"Base Vec: $base_vec, Strides: ${strides.drop(num_loop_dimensions)}")
//
//          val base_index = (base_vec zip strides.drop(num_loop_dimensions)) map { case (b, s) => b * s.to[ParameterIndex] } reduceTree { _ + _ }
//
//            // Get flat index for each (corner + origin)
//            val indices: Seq[ParameterIndex] = corners map {
//              corner =>
//                val offset = ((corner zip parallel_strides) map {
//                  case (cc, stride) =>
//                    cc * stride
//                }).sum.to[ParameterIndex]
//                base_index + base + offset
//            }
//
//            // Get weighted sum
//            hypervolumes.map(_.to[OutputType]).zip(indices).map {
//              case (hv, i) =>
//                hv * params(i, unit)
//            }.reduceTree {
//              _ + _
//            }
//          } else {
//            val input_staged = expanded_arg(Seq(batch, unit, I32(current_dimension)), new_enable)
//            // finish recursively.
//            Pipe.Reduce(Reg[OutputType](0))(2 by 1) {
//              bit =>
//                val input = input_staged()
//                val floored = floor(input).to[AccumResidualType]
//                val diff = input - floored
//                val residual_pair = scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
//                val beqz = bit.infix_==(I32(0))
//                val step = mux(beqz, 0.to[ParameterIndex], strides(current_dimension).to[ParameterIndex])
//                // if bk == 0 then we take the weight to be 1-xk otherwise we use xk.
//                val weight = mux(beqz, residual_pair(1), residual_pair(0))
//                val recursive = recursive_fill(current_index :+ bit, base + step)
//                recursive * weight.to[OutputType]
//            } {
//              _ + _
//            }
//          }
//
//          checkpoint(s"LatticeLoop${current_dimension}_end")
//          result
//        }
//        val reg = Reg[OutputType]
//        reg := recursive_fill(scala.Seq.empty[ParameterIndex], argon.uconst[ParameterIndex](0))
//        checkpoint("PostRecursive")
//
//        () => {
//          reg
//        }
//      }

      // A lattice goes from (batch, dim, unit) -> (batch, unit)
      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(units))
    }
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

    private def combine[T](xs: Seq[Seq[T]], func: (T, T) => T): Seq[Seq[T]] = {
      xs.length match {
        case 0 => throw new Exception("Empty reduction level:")
        case 1 => xs
        case len if len % 2 == 0 => combine(List.tabulate(len / 2) { i => join(xs(2 * i), xs(2 * i + 1), func) }, func)
        case len => combine(List.tabulate(len / 2) { i => join(xs(2 * i), xs(2 * i + 1), func) } :+ xs.last, func)
      }
    }
  }

}
