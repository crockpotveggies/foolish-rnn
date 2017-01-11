package ai.bernie.train;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.stats.impl.DefaultStatsUpdateConfiguration;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

/**
 * This example trains a RNN. Using the Lotto 649 dataset, the RNN
 * will start to predict sequential numbers that resemble "winning"
 * lottery numbers.
 *
 * Partly derived from the Shakespeare character generator.
 *
 * @author crockpotveggies
 */
public class RNNFool {
  private static final Logger logger = LoggerFactory.getLogger(RNNFool.class);

  public static void main(String[] args) throws Exception {
    logger.info("Initializing...");
    int lstmLayerSize = 50;
    int lstmLayerSize2 = 50;
    int lstmLayerSize3 = 50;
    int lstmLayerSize4 = 50;
    int miniBatchSize = 32;						//Size of mini batch to use when  training
    int exampleLength = 700;					//Length of each training example sequence to use. This could certainly be increased
    int tbpttLength = 7;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    int numEpochs = 9;							//Total number of training epochs
    int generateSamplesEveryNMinibatches = 100;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
    int nCharactersToSample = 6;				//Length of each sample to generate
    String generationInitialization = "\n";		//Optional character initialization; a random character is used if null
    Random rng = new Random(12345);

    // ui server
    final StatsStorage statsStorage = new InMemoryStatsStorage();
    final UIServer uiServer = UIServer.getInstance();
    uiServer.attach(statsStorage);


    logger.info("\nLoading training data...");
    CharacterIterator trainIter = getShakespeareIterator(miniBatchSize, 6);
    int nOut = trainIter.totalOutcomes();
    System.out.println("Total input columns: "+trainIter.inputColumns());
    System.out.println("Total outcomes: "+nOut);


    logger.info("Building network...");
    //Set up network configuration:
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
        .learningRate(0.3)
        .rmsDecay(0.96)
        .seed(12345)
        .regularization(true)
        .l1(0.001)
        .l2(0.0001)
        .dropOut(0.7)
        .weightInit(WeightInit.XAVIER)
        .updater(Updater.RMSPROP)
        .activation(Activation.TANH)
        .list()
        .layer(0, new GravesLSTM.Builder().nIn(trainIter.inputColumns()).nOut(lstmLayerSize).build())
        .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize2).build())
        .layer(2, new GravesLSTM.Builder().nIn(lstmLayerSize2).nOut(lstmLayerSize3).build())
        .layer(3, new GravesLSTM.Builder().nIn(lstmLayerSize3).nOut(lstmLayerSize4).build())
//        .layer(4, new GravesLSTM.Builder().nIn(lstmLayerSize4).nOut(lstmLayerSize3).build())
//        .layer(5, new GravesLSTM.Builder().nIn(lstmLayerSize3).nOut(lstmLayerSize2)
//            .activation("tanh").build())
        .layer(4, new RnnOutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
            .nIn(lstmLayerSize4).nOut(nOut).build())
        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
        .pretrain(false).backprop(true)
        .build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(
        new ScoreIterationListener(1),
        new StatsListener(statsStorage)
    );

    //Print the  number of parameters in the network (and for each layer)
    Layer[] layers = net.getLayers();
    int totalNumParams = 0;
    for( int i=0; i<layers.length; i++ ){
      int nParams = layers[i].numParams();
      System.out.println("Number of parameters in layer " + i + ": " + nParams);
      totalNumParams += nParams;
    }
    logger.info("Total number of network parameters: " + totalNumParams);

    logger.info("Beginning training...");
    int miniBatchNumber = 0;
    for( int i=0; i<numEpochs; i++ ){
      while(trainIter.hasNext()){
        DataSet ds = trainIter.next();
        net.fit(ds);
        if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
          System.out.println("--------------------");
          System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters at epoch " + i );
          System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "1" : generationInitialization) + "\"");
          String[][] samples = sampleCharactersFromNetwork(generationInitialization,net,trainIter,rng,nCharactersToSample,nSamplesToGenerate);

          // evaluate quality of lottery number by checking
          // order and presence of \n
          double integrity = 0.0;
          for( int j=0; j<samples.length; j++ ){
            boolean passed = true;
            String prevNum = samples[j][0];
            if(prevNum=="\n") {
              passed = false;
              break;
            }

            for( int k=1; k<samples[j].length; k++ ) {
              if(samples[j][k]=="\n") {
                passed = false;
                break;
              }
              if(Integer.parseInt(prevNum) >= Integer.parseInt(samples[j][k])) {
                passed = false;
                break;
              }
              prevNum = samples[j][k];
            }
            if(passed) integrity += 1.0;
          }

          integrity = integrity / (double) nSamplesToGenerate;
          System.out.println("----- Recent sample had integrity of " + integrity + " -----");
          System.out.println("----- Net statistics | l1 " + net.calcL1(false) + " | l2 "+net.calcL2(false)+" -----");

          for( int j=0; j<samples.length; j++ ){
            System.out.print("\n");
            for( int k=0; k<samples[j].length; k++ ) {
              System.out.print(samples[j][k]+",");
            }
          }
          System.out.println();
        }
      }

      trainIter.reset();	//Reset iterator for another epoch
    }

    logger.info("Saving model...");
    ModelSerializer.writeModel(net, new File(System.getProperty("user.dir") + "RNNLottoGenerator.model"), true);
  }


  public static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength) throws Exception{
    //The Complete Works of William Shakespeare
    //5.3MB file in UTF-8 Encoding, ~5.4 million characters
    //https://www.gutenberg.org/ebooks/100
    String userDir = System.getProperty("user.dir");
    String fileLocation = userDir + "/data/649.txt";	//Storage location from downloaded file

    File f = new File(fileLocation);
    if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//Download problem?

    String[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
    return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
        miniBatchSize, sequenceLength, validCharacters, new Random(12345));
  }


  private static String[][] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                      CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
    //Set up initialization. If no initialization: use a random character
    if( initialization == null ){
      initialization = String.valueOf(iter.getRandomCharacter());
    }

    //Create input for initialization
    INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
    int idx = iter.convertCharacterToIndex(initialization);
    for( int j=0; j<numSamples; j++ ){
      initializationInput.putScalar(new int[]{j,idx,0}, 1.0f);
    }

    String[][] out = new String[numSamples][charactersToSample];

    for(int i = 0; i < numSamples; i++) {
      //Sample from network (and feed samples back into input) one character at a time (for all samples)
      //Sampling is done in parallel here
      if(i % 2 == 0) net.rnnClearPreviousState();
      INDArray output = net.rnnTimeStep(initializationInput);
      output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);  //Gets the last time step output

      for (int j = 0; j < charactersToSample; j++) {
        //Set up next input (single time step) by sampling from previous output
        INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
        //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
        double[] outputProbDistribution = new double[iter.totalOutcomes()];
        for (int k = 0; k < outputProbDistribution.length; k++) outputProbDistribution[k] = output.getDouble(0, k);
        int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

        nextInput.putScalar(new int[]{0, sampledCharacterIdx}, 1.0f);    //Prepare next time step input
        out[i][j] = iter.convertIndexToCharacter(sampledCharacterIdx);

        output = net.rnnTimeStep(nextInput);  //Do one time step of forward pass
      }
    }

    return out;
  }


  public static int sampleFromDistribution( double[] distribution, Random rng ){
    double d = rng.nextDouble();
    double sum = 0.0;
    for( int i=0; i<distribution.length; i++ ){
      sum += distribution[i];
      if( d <= sum ) return i;
    }
    //Should never happen if distribution is a valid probability distribution
    throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
  }

}
