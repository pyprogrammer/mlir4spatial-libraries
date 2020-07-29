package mlir_libraries

import spatial.libdsl._
import spatial.metadata.memory._

object CommonConstructs {

  private def computeStrides[T: Num](shape: Seq[T])(implicit state: argon.State): Seq[T] = {
    val strides = shape.scanRight(1.to[T]) {
      case (s, stride) =>
        s * stride
    }
    strides.drop(1)
  }

  def Materialize[T: Num](arg: types.ReadableND[T])(implicit state: argon.State): types.ReadableND[T] = {
    val size = arg.shape reduceTree {
      _ * _
    }
    val intermediate = SRAM[T](size)
    val strides = computeStrides(arg.shape)

    val ctrs = arg.shape map { x => Counter.from(x by I32(1)) }

    Foreach(ctrs) {
      nd_index => {
        val index = utils.computeIndex(nd_index, strides)
        intermediate(index) = arg(nd_index: _*)()
      }
    }

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      override def apply(index: spatial.dsl.I32*): () => T = {
        () => {
          val ind = utils.computeIndex(index, strides)
          intermediate(ind)
        }
      }
    }
  }

  def LazyMaterialize[T: Num](arg: types.ReadableND[T], par_factors: Option[IndexedSeq[Int]] = Some(IndexedSeq.empty[Int]))(implicit state: argon.State): types.ReadableND[T] = {
    val size = arg.shape reduceTree {
      _ * _
    }
    val strides = computeStrides(arg.shape)
    val valid = SRAM[Bit](size).nonbuffer
    val intermediate = SRAM[T](size).nonbuffer

    val par = par_factors match {
      case Some(pf) => pf
      case None => arg.shape map { _ => 1}
    }

    Foreach(size by I32(1)) { i => valid(i) = 0.to[Bit] }

//    val dim = arg.shape.size
//    val input_fifo = Range(0, dim, 1) map { _ => FIFO[I32](I32(32)) }

//    println(input_fifo.mkString(", "))

    val ctrs = (arg.shape zip par) map { case (s, p) => Counter.from(s by I32(1) par I32(p)) }

    Pipe.Foreach(ctrs) {
      nd_index => {
        val index = utils.computeIndex(nd_index, strides)
        intermediate(index) = arg(nd_index: _*)()
        retimeGate()
        valid(index) = 1.to[Bit]
      }
    }

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      override def apply(index: I32*): () => T = {
        () => {
          val query_index = utils.computeIndex(index, strides)
          val is_valid = valid(query_index)

          ifThenElse[T](
            is_valid,
            () => {
              // Value is already valid
              intermediate(query_index)
            },
            () => {
              // Value isn't populated yet
              val break = Reg[Bit](false)

              Sequential(breakWhen = break).Foreach(*) {
                _ =>
                  break := valid(query_index)
              }

              intermediate(query_index)
            }
          )
        }
      }
    }
  }

  private def MaybeRead[T: argon.Type](fifo: FIFO[T], reg: Reg[T])(implicit state: argon.State, srcCtx: SrcCtx) = {
    ifThenElse(
      fifo.isEmpty,
      () => {},
      () => {
        reg := fifo.deq()
      }
    )
  }

  def LazyMaterialize2[T: Num](arg: types.ReadableND[T])(implicit state: argon.State, coprocessorScope: CoprocessorScope): types.ReadableND[T] = {
    val size = arg.shape reduceTree {
      _ * _
    }
    val strides = computeStrides(arg.shape)
    val valid = SRAM[Bit](size).nonbuffer.dontTouch
    valid.explicitName = "materialize_valid"
    val enqueued = SRAM[Bit](size).nonbuffer.dontTouch
    enqueued.explicitName = "materialize_enqueued"
    val intermediate = SRAM[T](size).nonbuffer.dontTouch
    intermediate.explicitName = "materialize_intermediate"

    val dim = arg.shape.size
    val input_fifo = Range(0, dim, 1) map { _ => FIFO[I32](I32(32)) }

    coprocessorScope.register(
      command_fifo => {
        Sequential {
          Parallel {
            Foreach(size by I32(1)){ i => valid(i) = 0.to[Bit] }
            Foreach(size by I32(1)) { i => enqueued(i) = 0.to[Bit] }
          }

          val b = Reg[Bit](false, "breakwhen")
          val last = Reg[T](0.to[T], "last_value").dontTouch
          val dbg_index = Reg[I32](-1.to[I32], "dbg_index").dontTouch

          Stream(breakWhen = b)(implicitly[SrcCtx], state).Foreach(*) {
            _ =>
              Pipe {
                utils.checkpoint("Stream")
                val not_ready = input_fifo map {
                  _.isEmpty
                } reduceTree {
                  _ || _
                }
                ifThenElse(not_ready, () => {
                  // Maybe we're done?
                }, () => {
                  val nd_index = input_fifo map {
                    _.deq
                  }
                  val index = utils.computeIndex(nd_index, strides)
                  dbg_index := index
                  utils.checkpoint("PostDeq")
                  ifThenElse(
                    enqueued(index), () => {}, () => {
                      enqueued(index) = 1.to[Bit]
                      val result = arg(nd_index: _*)()
                      intermediate(index) = result
                      last := result
                      retimeGate()
                      valid(index) = 1.to[Bit]
                    }
                  )
                })

                MaybeRead(command_fifo, b)
              }
          }
          utils.checkpoint("StreamFinish")

        }
      }
    )

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      override def apply(index: I32*): () => T = {
        () => {
          val query_index = utils.computeIndex(index, strides)
          val is_valid = valid(query_index)

          utils.checkpoint("ApplyEntry")

          ifThenElse[T](
            is_valid,
            () => {
              // Value is already valid
              intermediate(query_index)
            },
            () =>
            {
              val result: Reg[T] = Reg[T](Bits[T].zero, "MaterializeResult").dontTouch
              Pipe {
                // Value isn't populated yet
                val check2 = utils.checkpoint("Unpopulated")

                Parallel {
                  (index zip input_fifo) map {
                    case (ind, fifo) => fifo.enq(ind)
                  }
                }

                val check3 = utils.checkpoint("PostEnq")

                retimeGate()

                val break = Reg[Bit](false)
                Sequential(breakWhen = break).Foreach(*) {
                  _ =>
                    break := valid(query_index)
                }

                retimeGate()

                val check4 = utils.checkpoint("ValueReady")

                result := intermediate(query_index)
              }
              result.value
            }
          )
        }
      }
    }
  }
}
