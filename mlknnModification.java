package com.aarthy.miniproject;
import java.io.BufferedReader;
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


public class mlknnModification {
	
   public static void main(String[] args) throws Exception
   {
	  
  	   Instances fuzzyData=new Instances(new BufferedReader(new FileReader("assets//fuzzydata.arff")));
  	 Instances testData=new Instances(new BufferedReader(new FileReader("assets//emotions-test.arff")));
  	 Instances newData = new Instances(testData);
  	   for(int i=0;i<testData.numInstances();i++)
  	   {
  		   double sortedArray[]=new double[10];
  		   Instance instance = testData.instance(i);
  		   Instance newInstance = newData.instance(i);
  		   double a[][]=new double[fuzzyData.numInstances()][2];
  		   for(int j=0;j<fuzzyData.numInstances();j++)
  		   {    
  			   Instance instance1 = fuzzyData.instance(j);
  		       a[j][0]=j;
  		       a[j][1]=getDistance(instance,instance1);
  		     
  		   }
  		   sortedArray=sortArray(a);
  		   double avg[]=new double[6];
  		   double sum[]=new double[6];
	   
		   for(int k=0;k<sortedArray.length;k++)
	      	{ 
	    		Instance instance2=fuzzyData.get((int)sortedArray[k]);
	    		for(int label=0;label<6;label++)
	    		{
	    			sum[label]+=instance2.value(72+label);
	    		}
	    		
	      	}
		   for(int label=0;label<6;label++)
		   {
			   avg[label]=sum[label]/10;
			   if(avg[label]>=0.5)
				   newInstance.setValue(72+label, 1);
			   else
				   newInstance.setValue(72+label, 0);
		   }
  	   }
	 ArffSaver saver = new ArffSaver();
	 saver.setInstances(newData);
	 saver.setFile(new File("assets//newdata.arff"));
	 saver.writeBatch();
	 System.out.println("Saved to newData.arff");
    
   }
   public static double getDistance(Instance a, Instance b)
   {  
	   double result=9999999;
	   double sum=0;
	   for(int i=0;i<a.numAttributes()-6;i++)
	   {
		   sum=sum+Math.pow((a.value(i)-b.value(i)),2);
	   }
	   result = Math.sqrt(sum);
	   return result;	   
   }
   public static double[] sortArray(double a[][])
   {
	   for(int i=0;i<a.length;i++)
	   {
		   double temp1,temp2;
		   temp1=temp2=0;
		   for(int j=i;j<a.length-1;j++)
		   {
			   if(a[j][1]>a[j+1][1])
			   {
				   temp1=a[i][0];
				   temp2=a[i][1];
				   a[i][0]=a[j][0];
				   a[i][1]=a[j][1];
				   a[j][0]=temp1;
				   a[j][1]=temp2;
				   
			   }
		   }
	   }
	   double results[]=new double[10];
	   for(int k=0;k<10;k++)
	   {
		   results[k]=a[k][1];   
	   }
	   return results;
   }
   
   
   
}
