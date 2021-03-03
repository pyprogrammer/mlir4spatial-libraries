package tensorflow_lattice
import mlir_libraries.types._
import mlir_libraries.utils.checkpoint
import spatial.libdsl._
import mlir_libraries.{Coprocessor, CoprocessorScope, Tensor => MLTensor}
import _root_.spatial.libdsl
import _root_.spatial.metadata.memory._
import _root_.spatial.node.ForeverNew
import argon.stage
import mlir_libraries.debug_utils.TagVector

trait LatticeBase[U <: LatticeBase[U]] {
  type LT = U

  val shape: MLTensor[Int]
  val lattice_kernel: MLTensor[Double]
  val units: Int
  val PO2Opt: Boolean

  def dimensions = shape.shape.head
  def num_loop_dimensions: Int
  def parallel_dimensions = dimensions - num_loop_dimensions
  def strides = mlir_libraries.utils.ComputeStrides(shape.flatten.toIndexedSeq)
  def parallel_strides = strides.drop(num_loop_dimensions)

  // needed to pass shape into readable def
  def lattice_shape = shape

  // Get all vertices of hypercube and reverse so that these are opposite the hypervolumes
  lazy val corners: Seq[Seq[scala.Int]] = HypercubeLattice.allCorners(Seq.fill(parallel_dimensions)(1)).reverse

  // To be overridden by implementation
  def apply[T: libdsl.Num](arg: ReadableND[T])(implicit state: argon.State, coprocessorScope: CoprocessorScope): ReadableND[T]

  def getShape(inputShape: Seq[I32]): Seq[I32] = {
    // A lattice goes from (batch, dim, unit) -> (batch, unit)
    inputShape.dropRight(2) ++ Seq(I32(units))
  }
}

trait FullyUnrolledLattice extends LatticeBase[FullyUnrolledLattice] {

  override def apply[T: libdsl.Num](arg: ReadableND[T])(implicit state: argon.State, coprocessorScope: CoprocessorScope): ReadableND[T] = {

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis = 1)(arg)
    }
    assert(expanded_arg.shape.length == 3, "Expanded arg should have rank 3")

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }

    new ReadableND[T] {
      override def getInterface: Interface[T] = {
        val interfaces = Range(0, dimensions) map {
          _ => expanded_arg.getInterface
        }

        new Interface[T] {
          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            val batch = index(index.length - 2)
            val unit = index.last
            Range(0, dimensions) foreach {
              dim => interfaces(dim).enq(Seq(batch, unit, I32(dim)), ens)
            }
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            val batch = index(index.length - 2)
            val unit = index.last
            val params = LUT[OutputType](lattice_kernel.shape.head, units)(param_list: _*)

            val intermediate_values = Range(0, dimensions) map {
              dim =>
                interfaces(dim).deq(Seq(batch, unit, I32(dim)), ens)
            }

            mlir_libraries.debug_utils.TagVector("LatticeInputsTag", intermediate_values, ens)

            val parallelResidualPairs = intermediate_values map {
              value =>
                val floored = floor(value).to[AccumResidualType]
                val diff = value - floored
                scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
            }

            mlir_libraries.debug_utils.TagVector("parallelResidualPairs", parallelResidualPairs.flatten, ens)

            println(s"Parallel pairs: $parallelResidualPairs, length: ${parallelResidualPairs.length}")

            val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(parallelResidualPairs: _*)(_ * _)

            val base_vec = (Range(num_loop_dimensions, dimensions) zip intermediate_values) map {
              case (dim, inpt) =>
                (lattice_shape(dim), PO2Opt) match {
                  case (2, true) =>
                    0.to[ParameterIndex]
                  case (shape, _) =>
                    min(inpt.to[ParameterIndex], I32(shape - 2))
                }
            }

            println(s"Base Vec: $base_vec, Strides: ${strides.drop(num_loop_dimensions)}")

            val base_index = (base_vec zip strides) map { case (b, s) => b * s.to[ParameterIndex] } reduceTree { _ + _ }

            // Get flat index for each (corner + origin)
            val indices: Seq[ParameterIndex] = corners map {
              corner =>
                val offset = ((corner zip parallel_strides) map {
                  case (cc, stride) =>
                    cc * stride
                }).sum.to[ParameterIndex]
                base_index + offset
            }

            // Get weighted sum
            val v = hypervolumes.map(_.to[OutputType]).zip(indices).map {
              case (hv, i) =>
                hv * params(i, unit)
            }.reduceTree {
              _ + _
            }
            mlir_libraries.debug_utils.TagVector("LatticeOutput", Seq(v), ens)
            v
          }
        }
      }

      // A lattice goes from (batch, dim, unit) -> (batch, unit)
      lazy val shape: Seq[I32] = getShape(arg.shape)
    }
  }
}

trait ReduceBasedLattice extends LatticeBase[ReduceBasedLattice] {
  override def apply[T: libdsl.Num](arg: ReadableND[T])(implicit state: argon.State, coprocessorScope: CoprocessorScope): ReadableND[T] = {

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis = 1)(arg)
    }
    assert(expanded_arg.shape.length == 3, "Expanded arg should have rank 3")

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }

    new ReadableND[T] {
      override def getInterface: Interface[T] = {
        val interfaces = Range(0, dimensions) map {
          _ => expanded_arg.getInterface
        }

        new Interface[T] {
          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            val batch = index(index.length - 2)
            val unit = index.last
            Range(0, dimensions) foreach {
              dim => interfaces(dim).enq(Seq(batch, unit, I32(dim)), ens)
            }
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            val batch = index(index.length - 2)
            val unit = index.last
            val params = LUT[OutputType](lattice_kernel.shape.head, units)(param_list: _*)

            val intermediate_values = Range(0, dimensions) map {
              dim =>
                interfaces(dim).deq(Seq(batch, unit, I32(dim)), ens)
            }

            val sramReads = intermediate_values

            mlir_libraries.debug_utils.TagVector("LatticeInputsTag", sramReads, ens)

            def recursive_fill(current_index: scala.Seq[ParameterIndex], base: ParameterIndex): OutputType = {
              val current_dimension = current_index.size
              checkpoint(s"LatticeLoop${current_dimension}_begin")
              val result: OutputType = if (current_dimension == num_loop_dimensions) {
                // finish directly

                val remainingInputs = Range(num_loop_dimensions, dimensions) map {
                  dim => sramReads(dim)
                }

                val parallelResidualPairs = remainingInputs map {
                  value =>
                    val floored = floor(value).to[AccumResidualType]
                    val diff = value - floored
                    scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
                }

                mlir_libraries.debug_utils.TagVector("parallelResidualPairs", parallelResidualPairs.flatten, ens)

                println(s"Parallel pairs: $parallelResidualPairs, length: ${parallelResidualPairs.length}")

                val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(parallelResidualPairs: _*)(_ * _)

                val base_vec = (Range(num_loop_dimensions, dimensions) zip remainingInputs) map {
                  case (dim, inpt) =>
                    (lattice_shape(dim), PO2Opt) match {
                      case (2, true) =>
                        0.to[ParameterIndex]
                      case (shape, _) =>
                        min(inpt.to[ParameterIndex], I32(shape - 2))
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
                // finish recursively.
                Pipe.Reduce(Reg[OutputType](0))(2 by 1) {
                  bit =>
                    val input = sramReads(current_dimension)
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
            val v = recursive_fill(scala.Seq.empty[ParameterIndex], argon.uconst[ParameterIndex](0))
            mlir_libraries.debug_utils.TagVector("LatticeOutput", Seq(v), ens)
            v
          }
        }
      }

      // A lattice goes from (batch, dim, unit) -> (batch, unit)
      lazy val shape: Seq[I32] = getShape(arg.shape)
    }
  }
}

trait StreamReduceLattice extends LatticeBase[StreamReduceLattice] {

  val FIFODepth = 32

  override def apply[T: libdsl.Num](arg: ReadableND[T])(implicit state: argon.State, coprocessorScope: CoprocessorScope): ReadableND[T] = {

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis = 1)(arg)
    }
    assert(expanded_arg.shape.length == 3, "Expanded arg should have rank 3")

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }

    implicit val ev: Bits[Vec[T]] = Vec.fromSeq(Range(0, dimensions) map { _ => 0.to[T] })
    @struct case class InputBundle(unit: I32, input: Vec[T])
    @struct case class WithValid(valid: Bit, value: T)

    new ReadableND[T] {
      override def getInterface: Interface[T] = {
        val interfaces = Range(0, dimensions) map {
          _ => expanded_arg.getInterface
        }

        val indexFIFO = coprocessorScope.escape {
          implicit val ev: Bits[Vec[I32]] = Vec.fromSeq(Seq(I32(0), I32(0)))
          val tmp = FIFO[Vec[I32]](I32(FIFODepth))
          tmp.explicitName = "IndexFIFO"
          tmp
        }

        val valueFIFO = coprocessorScope.escape {
          val tmp = FIFO[InputBundle](I32(FIFODepth))
          tmp.explicitName = "valueFIFO"
          tmp
        }

        val busyValueFIFO = coprocessorScope.escape {
          val tmp = FIFO[InputBundle](I32(FIFODepth))
          tmp.explicitName = "busyValueFIFO"
          tmp
        }

        coprocessorScope.setup {
          'LatticeFetch.Pipe.Foreach(*) {
            i =>
              val index = indexFIFO.deq()
              println(s"Index Width: ${index.width}")
              val batch = index(index.width - 2)
              val unit = index(index.width - 1)

              val values = Range(0, dimensions) map {
                dim => interfaces(dim).deq(Seq(batch, unit, I32(dim)), Set(Bit(true)))
              }
              valueFIFO.enq(InputBundle(unit, Vec.fromSeq(values)))
          }
        }

        val intermediateResultFIFO = coprocessorScope.escape {
          val tmp = FIFO[T](I32(1 << num_loop_dimensions) * I32(4))
          tmp.explicitName = "IntermediateFIFO"
          tmp
        }

        coprocessorScope.setup {
          // first iteration we predicate the reads from values, otherwise we don't
          val valueRegisters = RegFile[T](I32(2), I32(dimensions))
          val unitRegister = RegFile[I32](I32(2))

          // Compute Counter Chain
          val counters = Seq(stage(ForeverNew())) ++ Seq.fill(num_loop_dimensions) {Counter.from(I32(2) by I32(1))}
          val params = LUT[OutputType](lattice_kernel.shape.head, units)(param_list: _*)

          'BusyStuffing.Pipe.Foreach(*) {
            _ =>
              busyValueFIFO.enq(InputBundle(I32(-1), Vec.fromSeq(Range(0, dimensions) map { _ => 0.to[T] })))
          }

          'LatticeCompute.Pipe.II(1).Foreach(counters) {
            iter =>
              val i = iter.head
              val index = iter.tail
              // are we on a fresh iteration?
              val shouldTrigger = spatial.dsl.ForcedLatency(0.0) { index.map {x => x === I32(0)}.reduceTree {_ & _} }
              val deqed = priorityDeq(List(valueFIFO, busyValueFIFO), List(shouldTrigger, shouldTrigger))
              val deqValues: Vec[T] = deqed.input
              val unitValue: I32 = deqed.unit

              // Bounce between two copies of the value registers
              val iterParity = i & I32(1)
              // Perform write into regfile
              Range(0, dimensions) foreach {
                dim =>
                  stage(spatial.node.RegFileWrite(valueRegisters, deqValues(I32(dim)), Seq(iterParity, I32(dim)), Set(shouldTrigger)))
              }
              stage(spatial.node.RegFileWrite(unitRegister, unitValue, Seq(iterParity), Set(shouldTrigger)))

              val values = Range(0, dimensions) map {
                dim => mux(shouldTrigger, deqValues(dim), valueRegisters(iterParity, I32(dim)))
              }

              val unit = mux(shouldTrigger, unitValue, unitRegister(iterParity))

              // compute one stage of the result, and then enqueue it. Someone else can handle the reduce.
              val remainingInputs = Range(num_loop_dimensions, dimensions) map {
                dim => values(dim)
              }

              val parallelResidualPairs = remainingInputs map {
                value =>
                  val floored = floor(value).to[AccumResidualType]
                  val diff = value - floored
                  scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
              }

              // compute the base vector for indexing the parameter space.
              val base_vec = values.zipWithIndex map {
                case (inpt, dim) =>
                  (lattice_shape(dim), PO2Opt) match {
                    case (2, true) =>
                      0.to[ParameterIndex]
                    case (shape, _) =>
                      min(inpt.to[ParameterIndex], I32(shape - 2))
                  }
              }

              // Components of base index from input
              val base_index_components = (base_vec zip strides) map {
                case (b, s) => b * s.to[ParameterIndex]
              }
              // components of base index from bumping
              val bump_components = (index zip strides) map {
                case (i, s) => i * s.to[ParameterIndex]
              }

              val base_index = (base_index_components ++ bump_components) reduceTree {_ + _}

              val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(parallelResidualPairs: _*)(_ * _)

              // Get flat index for each (corner + origin)
              val indices: Seq[ParameterIndex] = corners map {
                corner =>
                  val offset = ((corner zip parallel_strides) map {
                    case (cc, stride) =>
                      cc * stride
                  }).sum.to[ParameterIndex]
                  base_index + offset
              }

              // Get weighted sum
              val intermediateResult = hypervolumes.map(_.to[OutputType]).zip(indices).map {
                case (hv, i) =>
                  hv * params(i, unit)
              }.reduceTree {
                _ + _
              }

              // Compute weight of intermediate result
              val weight = (index zip values) map {
                case (ind, value) =>
                  mux(ind === I32(0), floor(value + 1.to[T]) - value, value - floor(value))
              } reduceTree {_ * _}

              intermediateResultFIFO.enq(intermediateResult * weight, unit !== I32(-1))
          }
        }

        val outputFIFO = coprocessorScope.escape {
          val tmp = FIFO[T](I32(FIFODepth))
          tmp.explicitName = "LatticeOutputFIFO"
          tmp
        }

        def accumLoop[U](exp: => U) = {
          argon.withFlow("AccumLoop", s => {s.iterDiff = 1}){exp}
        }

        coprocessorScope.setup {
          'LatticeAccum.Pipe.Foreach(*) {
            _ =>
              val v = intermediateResultFIFO.deqVec(1 << num_loop_dimensions)
              val sum = Range(0, 1 << num_loop_dimensions) map { i => v(i) } reduceTree {_ + _}
              outputFIFO.enq(sum)
          }
        }

        new Interface[T] {
          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            val batch = index(index.length - 2)
            val unit = index.last
            Range(0, dimensions) foreach {
              dim => interfaces(dim).enq(Seq(batch, unit, I32(dim)), ens)
            }
            indexFIFO.enq(Vec.fromSeq(index), ens.toSeq.reduceTree {_ && _})
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            stage(spatial.node.FIFODeq(outputFIFO,ens))
          }
        }
      }

      // A lattice goes from (batch, dim, unit) -> (batch, unit)
      lazy val shape: Seq[I32] = getShape(arg.shape)
    }
  }
}

trait CollapsedReduceBasedLattice extends LatticeBase[CollapsedReduceBasedLattice] {
  override def apply[T: libdsl.Num](arg: ReadableND[T])(implicit state: argon.State, coprocessorScope: CoprocessorScope): ReadableND[T] = {

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis = 1)(arg)
    }
    assert(expanded_arg.shape.length == 3, "Expanded arg should have rank 3")

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }

    new ReadableND[T] {
      override def getInterface: Interface[T] = {
        val interfaces = Range(0, dimensions) map {
          _ => expanded_arg.getInterface
        }

        new Interface[T] {
          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            val batch = index(index.length - 2)
            val unit = index.last
            Range(0, dimensions) foreach {
              dim => interfaces(dim).enq(Seq(batch, unit, I32(dim)), ens)
            }
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            val batch = index(index.length - 2)
            val unit = index.last
            val params = LUT[OutputType](lattice_kernel.shape.head, units)(param_list: _*)

            val sramReads = Range(0, dimensions) map {
              dim =>
                interfaces(dim).deq(Seq(batch, unit, I32(dim)), ens)
            }

            val chain = Range(0, num_loop_dimensions) map {_ => Counter.from(2 by 1)}

            val result = Reg[T]
            Reduce(result)(chain) {
              ind =>
                println(s"Index Var: $ind")
                val remainingInputs = sramReads.drop(num_loop_dimensions)

                val parallelResidualPairs = remainingInputs map {
                  value =>
                    val floored = floor(value).to[AccumResidualType]
                    val diff = value - floored
                    scala.Seq(diff, 1.toFloat.to[AccumResidualType] - diff)
                }

                val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(parallelResidualPairs: _*)(_ * _)

                val base_vec = sramReads.zipWithIndex map {
                  case (inpt, dim) =>
                    (lattice_shape(dim), PO2Opt) match {
                      case (2, true) =>
                        0.to[ParameterIndex]
                      case (shape, _) =>
                        min(inpt.to[ParameterIndex], I32(shape - 2))
                    }
                }

                println(s"Base Vec: ${base_vec}")

                val base_index = (base_vec zip strides) map { case (b, s) => b * s.to[ParameterIndex] } reduceTree { _ + _ }

                val bump = ind zip strides map {
                  case (i, stride) =>
                    mux(i === I32(0), I32(0), I32(stride))
                } reduceTree {_ + _}

                println(s"Base Index: $base_index")

                // Get flat index for each (corner + origin)
                val indices: Seq[ParameterIndex] = corners map {
                  corner =>
                    val offset = ((corner zip parallel_strides) map {
                      case (cc, stride) =>
                        cc * stride
                    }).sum.to[ParameterIndex]
                    base_index + offset + bump
                }

                println(s"Indices: $indices")

                val weight = sramReads.take(num_loop_dimensions) zip ind map {
                  case (inpt, bit) =>
                    val diff = inpt - floor(inpt)
                      mux(bit===I32(0), 1.toFloat.to[OutputType] - diff.to[OutputType], diff.to[OutputType])
                } reduceTree {_ * _}

                // Get weighted sum
                val v = hypervolumes.map(_.to[OutputType]).zip(indices).map {
                  case (hv, i) =>
                    hv * params(i, unit)
                }.reduceTree {
                  _ + _
                }
                v * weight
            } {_ + _}
            result
          }
        }
      }

      // A lattice goes from (batch, dim, unit) -> (batch, unit)
      lazy val shape: Seq[I32] = getShape(arg.shape)
    }
  }
}
