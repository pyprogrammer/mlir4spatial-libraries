package neural_nets

import argon.uconst
import emul.FixedPoint
import spatial.libdsl._

sealed trait ActivationType
object RELU extends ActivationType
object Sigmoid extends ActivationType

class LayerForward[IntBits <: INT[_], MantissaBits <: INT[_], Sign <: BOOL[_]]
(val dims_in: Int, val dims_out: Int)(implicit state: argon.State, c1: Cast[Int, FixPt[Sign, IntBits, MantissaBits]]) {
  type T = FixPt[Sign, IntBits, MantissaBits]

  def evaluate_linear(input: Seq[T],
               params: ((scala.Int,scala.Int) => T),
               bias:   (scala.Int => T)): Seq[T] = {
    val output_data = Seq.tabulate(dims_out) {i => 
      val weights = Seq.tabulate(dims_in){j => params(i,j)}
      val wx = input.zip(weights).map{case (a,w) => a*w}.reduceTree {_+_}
      wx + bias(i)
    }
    output_data
  }

  def evaluate_act(input: Seq[T], fcn: (T => T)): Seq[T] = {
    Seq.tabulate(dims_out) {i => fcn(input(i))}
  }


}

protected object Defs {
  // val layers_dims = List(10,2,1)//List(12288, 20, 7, 5, 1)

  def choose_act(layer: Int, max: Int, relu: (T => T), sigmoid: (T => T) ): (T => T) = if (layer < max-1) relu else sigmoid
  def choose_inp(layer: Int, in1: (scala.Int => T), in2: (scala.Int => T) ): (scala.Int => T) = if (layer == 0) in1 else in2
  def choose_store(layer: Int, max: Int, func: ( () => Unit)): ( () => Unit) = { if (layer == max) func else {() => ()} }

  type intBits = _4
  val intBits = 4
  type mantissaBits = _28
  val mantissaBits = 28


  val totalBits = intBits + mantissaBits

  type T = FixPt[TRUE, intBits, mantissaBits]
}

@spatial object FCNN extends SpatialApp {
  import Defs._
  import spatial.dsl._

  def linear(x: Seq[T]): Seq[T] = x
  def relu(x: Seq[T]): Seq[T] = x.map{y => mux(y < 0, 0, y)}
  def sigmoid(x: Seq[T]): Seq[T] = {
    val numPoints = 256
    val stepSize = 0.1
    val data = List.tabulate[T](numPoints){i => 1/(1+scala.math.exp(-i*stepSize))}
    val sigLUT = LUT[T](numPoints)(data:_*)
    x.map{y => 
      val bin = (abs(y) / stepSize.to[T]).to[Int]
      val result = sigLUT(mux(bin > numPoints-1, numPoints-1, bin))
      mux(y < 0, {1-result}, result)
    }
  }
  def hard_sigmoid(x: Seq[T]): Seq[T] = {
    x.map{y => mux(y < -2.5.to[T], 0, mux(y > 2.5.to[T], 1, 0.2.to[T] * y + 0.5.to[T]))}
  }

  def linear_act(inputs: Seq[T], weights: Seq[Seq[T]], biases: Seq[T]): Seq[T] = {
    weights.zip(biases).map{case(w,b) => inputs.zip(w).map{case(a,b) => a*b}.reduceTree{_+_} + b}
  }

  def loop_evaluate_layer(input: SRAM1[T],
                        params: SRAM2[T],
                        bias: SRAM1[T],
                        output: SRAM1[T],
                        fcn: (Seq[T] => Seq[T]),
                        op: scala.Int,
                        ip: scala.Int
  ): Unit = {
    Foreach(output.length by 1 par op){ i => 
      output(i) = fcn(Seq(Reduce(Reg[T])(input.length by 1 par ip){ j => 
        input(j) * params(i,j) + bias(i)
      }{_+_})).head
    }
  }

  def main(args: Array[String]): Unit = {
    val layers_dims = List(41, 128,128,64,64,64,64,64,64,32,32, 2) //loadCSVNow[scala.Int](s"${System.getProperty("user.dir")}/tf-src/dnn-nids/DNN-NIDS_LUTs/DIMS.csv", ","){_.toInt}
    val input = (0::10000, 0::41){(i,j) => random[T]} //loadCSV2D[T](s"${System.getProperty("user.dir")}/tf-src/dnn-nids/DNN-NIDS_LUTs/INPUT_LUT.csv")
    val total_samples = input.rows // Transposed so single image is along leading dimension in memory
    val parameters = Seq.tabulate(layers_dims.size - 1){i => 
      (0::layers_dims(i),0::layers_dims(i+1)){(j,k) => random[T]}
      // loadCSV2D[T](s"${System.getProperty("user.dir")}/tf-src/dnn-nids/DNN-NIDS_LUTs/L${i+1}_NEURON_W_LUT.csv")
    }
    val biases = Seq.tabulate(layers_dims.size - 1){i => 
      Array.tabulate(layers_dims(i+1)){_ => random[T]}
      // loadCSV1D[T](s"${System.getProperty("user.dir")}/tf-src/dnn-nids/DNN-NIDS_LUTs/L${i+1}_NEURON_B_LUT.csv")
    }

    val input_DRAM = DRAM[T](input.rows, input.cols)
    val parameter_DRAMs = Seq.tabulate(layers_dims.size - 1){i => DRAM[T](parameters(i).rows, parameters(i).cols)}
    val bias_DRAMs = Seq.tabulate(layers_dims.size - 1){i => DRAM[T](biases(i).length)}

    setMem(input_DRAM, input)
    parameter_DRAMs.zip(parameters).foreach{case (dram, p) => setMem(dram,p)}
    bias_DRAMs.zip(biases).foreach{case (dram, p) => setMem(dram,p)}

    val output_DRAM = DRAM[T](total_samples,layers_dims.last)

    val points = ArgIn[Int]
    val pointsToDo = args(0).to[Int]
    println(r"Doing ${pointsToDo} points (max possible ${total_samples}")
    setArg(points, pointsToDo)

    val actuallyLoad = ArgIn[Int]
    val actuallyUseWeightsAndBiases = args(1).to[Int]
    setArg(actuallyLoad, actuallyUseWeightsAndBiases)

    Accel {
      val param_srams = Seq.tabulate(layers_dims.size-1){i => SRAM[T](I32(layers_dims(i+1)), I32(layers_dims(i))).fullfission}
      val bias_srams = Seq.tabulate(layers_dims.size-1){i => SRAM[T](I32(layers_dims(i+1))).fullfission}
      if (actuallyLoad == 1) {
        Parallel {
          param_srams.zip(parameter_DRAMs).foreach{case (sram, dram) => sram load dram}
          bias_srams.zip(bias_DRAMs).foreach{case (sram, dram) => sram load dram}
        }
      }

      Foreach (points par 1) { p => 
        val input_sram = SRAM[T](I32(layers_dims.head)).fullfission
        val output_sram = SRAM[T](I32(layers_dims.last)).fullfission
        val tmp_srams = List.tabulate(layers_dims.size-2){i => SRAM[T](layers_dims(i+1))}
        input_sram load input_DRAM(p,0 :: layers_dims.head par I32(512/totalBits))
        Pipe{
          loop_evaluate_layer(input_sram, param_srams(0), bias_srams(0), tmp_srams(0), x => relu(x), 8, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(0), param_srams(1), bias_srams(1), tmp_srams(1), x => relu(x), 4, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(1), param_srams(2), bias_srams(2), tmp_srams(2), x => relu(x), 4, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(2), param_srams(3), bias_srams(3), tmp_srams(3), x => relu(x), 2, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(3), param_srams(4), bias_srams(4), tmp_srams(4), x => relu(x), 2, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(4), param_srams(5), bias_srams(5), tmp_srams(5), x => relu(x), 1, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(5), param_srams(6), bias_srams(6), tmp_srams(6), x => relu(x), 1, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(6), param_srams(7), bias_srams(7), tmp_srams(7), x => relu(x), 1, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(7), param_srams(8), bias_srams(8), tmp_srams(8), x => relu(x), 1, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(8), param_srams(9), bias_srams(9), tmp_srams(9), x => relu(x), 1, 1)
        }

        Pipe{
          loop_evaluate_layer(tmp_srams(9), param_srams(10), bias_srams(10), output_sram, x => relu(x), 1, 1)
        }

        if (actuallyLoad == 1) output_DRAM(p,0::layers_dims.last par I32(512/totalBits)) store output_sram
      }
    }

    printMatrix(getMatrix(output_DRAM), "output")
    // val gold = loadCSV2D[T](s"${System.getProperty("user.dir")}/test_parameters/fcnn/output.csv")
    // printMatrix(gold, "Wanted")
  }
}

