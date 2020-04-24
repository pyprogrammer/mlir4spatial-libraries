package lattice

import spatial.libdsl._

@spatial object TestLatticeApp extends SpatialApp {

  import spatial.dsl._

  val sizes = scala.Array.fill(4)(2)
  val params_per_lattice = sizes.product
  val dimensions = sizes.length

  val num_lattices = 16
  val nparams = num_lattices * params_per_lattice

  type T = FixPt[TRUE, _3, _13]
  type O = FixPt[TRUE, _3, _13]

  val dir = System.getProperty("user.dir")

  val rtl_structure = Seq(
    17, 0, 33, 12,
    15, 7, 23, 6,
    5, 17, 23, 7,
    22, 14, 39, 3,
    40, 16, 20, 6,
    29, 7, 14, 0,
    13, 39, 40, 3,
    0, 38, 35, 31,
    7, 4, 16, 36,
    23, 30, 9, 21,
    29, 33, 9, 35,
    8, 32, 22, 7,
    24, 21, 39, 1,
    6, 40, 34, 9,
    22, 6, 13, 2,
    29, 3, 40, 37
  ).grouped(4)

  val calibrator_inputs = (0 to 11) map {
    i => i / 10.0
  }

  val calibrator_outputs = calibrator_inputs map {_ * 2}

  val calibrator_matches = Map(
    (0.1 -> 0.7), (0.2 -> 0.9)
  )

  override def main(args: Array[String]): Unit = {

    val iterations = ArgIn[Int]
    setArg(iterations, args(0).to[Int])

    // Just emulating the 41 inputs, padded up to make N * sizeof(T) to be a multiple of 512 bits.
    val input_DRAM = DRAM[T](64)
    val inputs = Array.tabulate[T](I32(64)) {
      i => 1.to[T]
    }
    setMem(input_DRAM, inputs)

    val o = ArgOut[T]

    Accel {
      val input = SRAM[T](64)
      input load input_DRAM

      val param = LUT.fromFile[O](num_lattices, params_per_lattice)(s"$dir/tf-src/LATTICE_NETWORK_16/LATTICE_PARAMS.csv")

      Foreach(iterations par 1) {
        _ =>
          // compute the calibrated results
          val calibrated_inputs = (0 to 40) map {
            //integer: Int, mantissa: Int, output: Int, sign: Boolean
            i => LatticeProgram.constructCalibrator(3, 13, 13, true,
              calibrator_inputs, calibrator_outputs, calibrator_matches
            ).uevaluate(input(I32(i)))
          }

          val result = (rtl_structure.zipWithIndex map {
            case (row, i) =>
              LatticeProgram.constructSimplex(3, 13, 13, true, sizes).uevaluate(
                {x: scala.Int => calibrated_inputs(row(x))},
                {x: I32 => param(I32(i), x)}
              ).asInstanceOf[O]
          }).toSeq reduceTree {_ + _}
          o := result
      }
    }

    println(o)
  }
}
