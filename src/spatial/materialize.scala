package mlir_libraries
import types._

import spatial.libdsl._
import _root_.spatial.{dsl, libdsl}
import _root_.spatial.metadata.memory._
import _root_.spatial.node.SRAMRead

// For MLIR-spatial native operations
trait Materialization {
  @forge.tags.api def Materialize[T: Num](parallelization: Int = 1, uptime: Fraction = Fraction(1, 1))(arg: types.ReadableND[T])
                         (implicit state: argon.State, cps: CoprocessorScope): types.ReadableND[T] = {
    if (Options.Coproc) {
      MaterializeCoproc(parallelization, uptime)(arg)
    } else {
      MaterializeSRAM(parallelization, uptime)(arg)
    }
  }

  private def computeStrides[T: Num](shape: Seq[T])(implicit state: argon.State): Seq[T] = {
    val strides = shape.scanRight(1.to[T]) {
      case (s, stride) =>
        s * stride
    }
    strides.drop(1)
  }

  private def gcd(a: Int, b: Int): Int = {
    BigInt(a).gcd(BigInt(b)).toInt
  }

  private def factorize(x: Int): List[Int] = {
    @scala.annotation.tailrec
    def foo(x: Int, a: Int = 2, list: List[Int] = Nil): List[Int] = a*a > x match {
      case false if x % a == 0 => foo(x / a, a    , a :: list)
      case false               => foo(x    , a + 1, list)
      case true                => x :: list
    }
    foo(x)
  }

  private def computeParFactors(shape: Seq[Int], parallelization: Int) = {
    // computes par factors for each dimension.
    val factors = shape flatMap factorize
    val factorization = (factors.sorted.groupBy(x => x) map {case (v, l) => (v, l.length)}).toSeq.sortBy(_._1).dropWhile(_._1 == 1)
    var bestPar = BigInt(Int.MaxValue)
    def process(counts: Seq[(Int, Int)], curPar: BigInt): Option[(Seq[Int], BigInt)] = {
      counts match {
        case (value, rep) +: rest =>
          val subresults = (0 to rep) flatMap {
            repetitions =>
              val newPar = curPar * BigInt(value).pow(repetitions)
              if (newPar >= parallelization) {
                if (newPar >= bestPar) {
                  // Already found a better solution.
                  Seq.empty
                } else {
                  // We have a better solution.
                  bestPar = newPar
                  // Fill remaining slots with 1
                  val solution = Seq(repetitions) ++ Seq.fill(rest.length)(0)
                  Seq((solution, newPar))
                }
              } else {
                // best match from the recursive allocation
                process(rest, newPar) match {
                  case Some(recursive) =>
                    if (recursive._2 < parallelization) {
                      // Insufficient parallelization
                      Seq.empty
                    } else {
                      if (recursive._2 <= bestPar) {
                        bestPar = recursive._2
                        Seq((Seq(repetitions) ++ recursive._1, recursive._2))
                      }
                      else {
                        Seq.empty
                      }
                    }
                  case None =>
                    Seq.empty
                }
              }
          }
          if (subresults.isEmpty) {
            None
          } else {
            Some(subresults.minBy(_._2))
          }
        case Nil =>
          if (curPar >= parallelization) {
            Some((Seq.empty, curPar))
          } else {
            None
          }
      }
    }

    process(factorization, 1) match {
      case Some((factorPowers, _)) =>
        var remainingPar = (((factorization map {x => BigInt(x._1)}) zip factorPowers) map {case (factor, power) => factor.pow(power)}).product
        shape map {
          part =>
            val alloc = remainingPar.gcd(part)
            remainingPar /= alloc
            alloc.toInt
        }
      case None =>
        println(s"Failed to find parallelization")
        // Couldn't reach the parallelization needed, so we instead fully parallelize.
        shape
    }
  }

  var materialization_cnt = -1
  def MaterializeSRAM[T: Num](parallelization: Int = 1, uptime: Fraction = Fraction(1, 1))(arg: types.ReadableND[T])(implicit state: argon.State): types.ReadableND[T] = {
    materialization_cnt += 1
    val materialization_capture = materialization_cnt

    val size = arg.shape reduceTree {
      _ * _
    }
    val intermediate = SRAM[T](size).nonbuffer
    intermediate.explicitName = f"materialization_sram_$materialization_capture"
    val strides = computeStrides(arg.shape)

    println(f"Arg shape: ${arg.shape.mkString(", ")}, par: ${parallelization}")

    // allocate parallelization factors, guaranteeing that the final par factor > requested parallelization.

    val shape = arg.shape flatMap {
      case argon.Const(s) => Seq(s.toInt)
      case _ => Seq.empty
    }

    val parFactors = computeParFactors(shape, parallelization)
    println(s"Requested Par: $parallelization, Recieved: $parFactors")
    assert(parFactors.length == shape.length)

    val ctrs = () => {
      val parIterator = parFactors.toIterator
      arg.shape map {
      case argon.Const(s) =>
        Counter.from(I32(s.toInt) by I32(1) par I32(parIterator.next()))
      case unk =>
        Counter.from(unk by I32(1))
    }}



    val interface = arg.getInterface

    val finishStream = Reg[Bit](false)
    finishStream.explicitName = "MaterializeCompletion"
    finishStream.nonbuffer.dontTouch
    finishStream := false
    Pipe {
      Pipe.Foreach(ctrs()) {
        nd_index => {
          interface.enq(nd_index, Set(Bit(true)))
        }
      }

      Pipe.Foreach(ctrs()) {
        nd_index => {
          val index = utils.computeIndex(nd_index, strides)
          intermediate(index) = interface.deq(nd_index, Set(Bit(true)))
        }
      }
      finishStream := true
    }

    new types.ReadableND[T] {
      override val shape: Seq[dsl.I32] = arg.shape

      override def getInterface: types.Interface[T] = {
        new types.Interface[T] {
          override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): Void = {}

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            // wait until finished
            val result = Reg[T]
            val ind = utils.computeIndex(index, strides)
            'SpinWaitOuter.Sequential {
              val break = Reg[Bit](false)
              'SpinWait.Sequential(breakWhen = break)(implicitly[SrcCtx], implicitly[argon.State]).Foreach(*) {
                _ =>
                  break := finishStream || !(ens.toSeq.reduceTree {_ && _})
              }
              result := argon.stage(SRAMRead(intermediate, Seq(ind), ens))
              mlir_libraries.debug_utils.TagVector("Index", index, ens)
              mlir_libraries.debug_utils.TagVector("Materializing", Seq(result.value), ens)
            }
            retimeGate()
            result.value
          }
        }
      }
    }
  }

  def MaterializeCoproc[T: Num](parallelization: scala.Int = 1, uptime: Fraction = Fraction(1, 1))(arg: types.ReadableND[T])(implicit state: argon.State, cps: CoprocessorScope, srcCtx: SrcCtx): types.ReadableND[T] = {

    val dimensions = arg.shape.length

    type CoprocessorLike = {
      def enq(inputs: Seq[I32], en: Bit): Void
      def deq(inputs: Seq[I32], en: Bit): T
    }

    val tmp = cps.escape { Vec.fromSeq(Range(0, dimensions) map {x => I32(0)}) }
    implicit val bitsEV: Bits[Vec[I32]] = tmp
    val coprocessors = {
      Range(0, parallelization) map { _ =>
        val subInterface = arg.getInterface
        new Coprocessor[Vec[I32], T] {
          override def coprocessorScope: CoprocessorScope = cps

          override def enq(inputs: Vec[I32]): Unit = {
            val stagedRegs = Range(0, dimensions) map {
              ind =>
//                  val reg = Reg[I32]
//                  reg := inputs(ind)
//                  reg.value
                inputs(ind)
            }
            subInterface.enq(stagedRegs, Set(Bit(true)))
          }

          override def deq(inputs: Vec[I32]): T = {
            val stagedRegs = Range(0, dimensions) map {
              ind =>
//                val reg = Reg[I32]
//                reg := inputs(ind)
//                reg.value
                inputs(ind)
            }
            subInterface.deq(stagedRegs, Set(Bit(true)))
          }
        }
      }
    }

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      var count = 0
      def getCount() = {
        val cnt = count
        count += 1
        cnt
      }

      override def getInterface: types.Interface[T] = {
        val coproc = coprocessors(getCount() % parallelization)
        val interface = coproc.interface
        new types.Interface[T] {
          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            interface.enq(Vec.fromSeq(index), ens.toSeq reduceTree {
              _ && _
            })
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            interface.deq(ens.toSeq reduceTree {
              _ && _
            })
          }
        }
      }
    }
  }
}
