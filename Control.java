package com.aarthy.miniproject;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;


public class Control {

    public static void main(String[] args) throws Exception {
    	
    	 MultiLabelInstances dataset = new MultiLabelInstances("assets//emotions-train.arff", "assets//emotions.xml");
    	 Instances data=dataset.getDataSet();

    	 FileReader reader = new FileReader("assets//emotions-test.arff");
         Instances unlabeledData = new Instances(reader);
         Instances fuzzifiedData = new Instances(unlabeledData);
    	 
    	 FileWriter writer = new FileWriter("output1.csv");
    	
         writer.write("");
         
       //MLkNN model = new MLkNN();
         //model.build(dataset);

         for(int k=0;k<6;k++)
         {
        	 data.setClassIndex(72+k);
             
             MultilayerPerceptron classifier = new MultilayerPerceptron();
             classifier.setLearningRate(0.1);
             classifier.buildClassifier(data);
        	 unlabeledData.setClassIndex(72+k);
             
             int numInstances = unlabeledData.numInstances();

             for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
                 Instance instance = unlabeledData.instance(instanceIndex);
                 Instance fuzzyData =fuzzifiedData.instance(instanceIndex);
                
                 double result=classifier.classifyInstance(instance);
                 
                 fuzzyData.setValue(72+k,result);
                 
                 writer.append(String.valueOf(result)+", ");
                 System.out.println(result);
             }
             writer.append("\n");
             System.out.println("END of "+k);
         }
         
         ArffSaver saver = new ArffSaver();
         saver.setInstances(fuzzifiedData);
         saver.setFile(new File("assets//fuzzydata.arff"));
         saver.writeBatch();
        
         writer.close();
         System.out.println("Saved.");
        

    }
}
