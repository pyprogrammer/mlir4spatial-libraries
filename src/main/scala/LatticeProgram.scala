package lattice

import scala.reflect.runtime.universe._
import scala.reflect.runtime.currentMirror
import scala.tools.reflect.ToolBox
import spatial.dsl.{spatial, SpatialApp}

object LatticeProgram {
  val toolbox = currentMirror.mkToolBox()

  val states = scala.collection.mutable.ArrayBuffer[argon.State]()


  def toType(b: Boolean): Tree = {
    if (b) tq"spatial.dsl.TRUE" else tq"spatial.dsl.FALSE"
  }

  def toType(i: Int): Tree = {
    tq"spatial.dsl.${TypeName("_" + i)}"
  }

  def constructFixPtType(sign: Boolean, integer: Int, fraction: Int): Tree = {
    tq"spatial.dsl.FixPt[${toType(sign)}, ${toType(integer)}, ${toType(fraction)}]"
  }

  def log2(x: Int): Int = {
    ((0 until x) find {1 << _ >= x}).get
  }

  def strides(dimensions: IndexedSeq[Int]): IndexedSeq[Int] = {
    val strides: Array[Int] = Array.fill(dimensions.length){1}
    (1 until dimensions.length) foreach {
      d => {
        strides(d) = strides(d-1) * dimensions(d-1)
      }
    }
    strides
  }

  def constructSimplex(integer: Int, mantissa: Int, output: Int, sign: Boolean, dimensions: Array[Int])
                      (implicit s: argon.State): SimplexLattice[_, _, _, _, _] = {
    val totalSize = dimensions.product
    val log2Size = log2(totalSize)
    val simplexLatticeType =
      tq"lattice.SimplexLattice[${toType(integer)}, ${toType(mantissa)}, ${toType(log2Size)}, ${toType(output)}, ${toType(sign)}]"



    val arrayargs = (strides(dimensions) map {d => Literal(Constant(d))}).toList

    val size = this.synchronized {
      states.append(s)
      states.size
    }

    val qq = q"""{
        import spatial.dsl._
        implicit val state: argon.State = lattice.LatticeProgram.states(${Literal(Constant(size - 1))})
        val n = new $simplexLatticeType(${Literal(Constant(dimensions.length))}, scala.Array(..$arrayargs))
        n
        }"""

    toolbox.eval(qq).asInstanceOf[SimplexLattice[_, _, _, _, _]]
  }

  def constructHypercube(integer: Int, mantissa: Int, output: Int, sign: Boolean, dimensions: Array[Int])
                      (implicit s: argon.State): HypercubeLattice[_, _, _, _, _] = {
    val totalSize = dimensions.product
    val log2Size = log2(totalSize)
    val simplexLatticeType =
      tq"lattice.HypercubeLattice[${toType(integer)}, ${toType(mantissa)}, ${toType(log2Size)}, ${toType(output)}, ${toType(sign)}]"



    val arrayargs = (strides(dimensions) map {d => Literal(Constant(d))}).toList

    val size = this.synchronized {
      states.append(s)
      states.size
    }

    val qq = q"""{
        import spatial.dsl._
        implicit val state: argon.State = lattice.LatticeProgram.states(${Literal(Constant(size - 1))})
        val n = new $simplexLatticeType(${Literal(Constant(dimensions.length))}, scala.Array(..$arrayargs))
        n
        }"""

    toolbox.eval(qq).asInstanceOf[HypercubeLattice[_, _, _, _, _]]
  }

  val calibrator_params = scala.collection.mutable.ArrayBuffer[(Seq[Double], Seq[Double], Map[Double, Double])]()
  def constructCalibrator(integer: Int, mantissa: Int, output: Int, sign: Boolean, input_breakpoints: Seq[Double],
                          output_values: Seq[Double],
                          exact_matches: Map[Double, Double])(implicit s: argon.State): Calibrator[_, _, _, _] = {
    val (calibrator_data, size) = this.synchronized {
      calibrator_params.append((input_breakpoints, output_values, exact_matches))
      states.append(s)
      (calibrator_params.size, states.size)
    }

    val tp = tq"lattice.Calibrator[${toType(integer)}, ${toType(mantissa)}, ${toType(output)}, ${toType(sign)}]"

    val qq =
      q"""{
         import spatial.dsl._
         implicit val state: argon.State = lattice.LatticeProgram.states(${Literal(Constant(size - 1))})
         val (input_breakpoints, output_values, exact_matches) =
            lattice.LatticeProgram.calibrator_params(${Literal(Constant(calibrator_data - 1))})
         val calibrator = new $tp(input_breakpoints, output_values, exact_matches)
         calibrator
         }"""

    toolbox.eval(qq).asInstanceOf[Calibrator[_, _, _, _]]
  }
}

// TODO (stanfurd): Implement a metaprogrammed general test.

@spatial object TestLatticeProgram extends SpatialApp {
  import spatial.dsl._

  val sizes = scala.Array.fill(8)(2)
  val dimensions = sizes.length

  type T = FixPt[FALSE, _3, _13]
  type O = FixPt[TRUE, _3, _13]
  override def main(args: Array[String]): Unit = {
    val out = ArgOut[O]


    val iterations = ArgIn[Int]
    setArg(iterations, args(0).to[Int])

    val input = loadCSV1D[T](s"${System.getProperty("user.dir")}/test_parameters/simplex/${dimensions}-${sizes(0)}/input.csv")
    val parameters = loadCSV1D[O](s"${System.getProperty("user.dir")}/test_parameters/simplex/${dimensions}-${sizes(0)}/parameters.csv")

    val input_DRAM = DRAM[T](dimensions)
    val parameter_DRAM = DRAM[O](sizes.product)

    setMem(input_DRAM, input)
    setMem(parameter_DRAM, parameters)
    Accel {
      val input_sram = SRAM[T](I32(dimensions)).fullfission
      val param_sram = SRAM[O](I32(sizes.product)).fullfission
      Parallel {
        input_sram load input_DRAM(0 :: dimensions par I32(8))
        param_sram load parameter_DRAM(0 :: sizes.product par I32(32))
      }
      val evaluator = lattice.LatticeProgram.constructSimplex(3, 13, 3, false, sizes)
      Foreach (iterations par 1) {
        _ => {
          out := evaluator.uevaluate(
            { x: scala.Int => input_sram(I32(x)) },
            { x: I32 => param_sram(x) }
          ).asInstanceOf[O]
        }
      }
    }
    println(r"Received Output: $out")
    val gold = loadCSV1D[T](s"${System.getProperty("user.dir")}/test_parameters/simplex/${dimensions}-${sizes(0)}/output.csv")
    println(r"Wanted: ${gold(0)}")
  }
}
