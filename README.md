# Domain Adaptation for Facial Emotion Recognition

# Abstract

Facial expression recognition is a challenging task in the domain of deep learning. Ensuring good performance is the cornerstone of the research problem related to transfer learning and domain adaptation. In the facial expression recognition task, a well-performing deep neural network model trained on one dataset (source domain) typically underperforms when subjected to a different but relevant dataset (target domain). This under-performance is the result of facial expression variation across different domains, even in the case of same expression. In our case, we have Western faces as source domain and Pakistani faces as target domain. Further problem arises in using Pakistani
faces is the lack of samples, and co-existence of different ethnicity in the region. We used input and feature space domain adaoptation approaches to address the problem of domain adaptation.
WGAN failed to produce expected results, whereas CycleGAN and feature-based domain adaptive models performed reasonably good on our data. We were able to get very close to the baseline results using domain adaptation approaches used and even got slightly better accuracy for VGG16 in one of the approach.

# Dataset

We Used RAF_DB as source dataset and for target dataset we collected and annotated Pakistani facial images from different internet sources.

<figure class="image"><img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/RAF_Pak.jpg" alt="System Diagram" style="display: block;  margin-left: auto;  margin-right: auto;"><figcaption></figcaption></figure>

## Source Dataset 

Real-world Affective Faces Database (RAF_DB) contains 15338 coloured images of westren facial expressions, we are using only seven classes although RAF_DB contains more than seven classes and around 30K images.

<img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/Source%20Dataset%20Stat%20upd.JPG" alt="Source Dataset" style="width:90%;display: block;  margin-left: auto;  margin-right: auto;">

## Target Dataset

We have collected and annotated around 4000 coloured Pakistani facial expression images as out target dataset.

<img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/Target%20Dataset%20Stat%20upd.JPG" alt="Target Dataset" style="width:90%;display: block;  margin-left: auto;  margin-right: auto;">

# Experiments and Results

We used following mentioned approaches to attempt domain adaptation problem:

1. Unsupervised Domain Adaptation using WGAN — WGAN results were not useable so,this approach was discontinued.
2. Semi-supervised Domain Adaptation using CycleGAN.
3. Feature Space Unsupervised Domain Adaptation.

We have used two target datasets in our experimentation. The first dataset
is used in domain adaptation process and second dataset is
kept unseen in all the ways for testing purposes. This was
done to ensure model performance consistency on target domain. We used two classifiers in our experimentation. First one
is VGG16 pre-trained on ImageNet Dataset and second is
ResNET18 pre-trained on ImageNet Dataset. 

## Baseline results

### Source Domain Dataset Results

We started ourn experimentation by training VGG16 and ResNet18 on source domain dataset and their accuracies on
source domain are provided below.

<table class="table table-bordered">
  <thead class="thead-dark">
    <tr>
      <th scope="col">Classifier</th>
      <th scope="col">Source Domain Training Accuracy</th>
      <th scope="col">Source Domain Testing Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VGG16</td>
      <td>94.8</td>
      <td>79.45</td>
    </tr>
    <tr>
      <td>ResNET18</td>
      <td>92.3</td>
      <td>80.17</td>
    </tr>
  </tbody>
</table>

### Target Domain Dataset Results

In our experimentation, we first evaluated our classifiers
(VGG16 and ResNET18) on target domain without doing
any kind of domain adaptation. The baseline results for the
classifiers used are provided in the following table:

<table class="table table-bordered">
  <thead class="thead-dark">
    <tr>
      <th scope="col">Classifier</th>
      <th scope="col">Target Dataset 1 Accuracy (Unseen)</th>
      <th scope="col">Target Dataset 2 Accuracy (Unseen)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VGG16</td>
      <td>50.92</td>
      <td>37.51</td>
    </tr>
    <tr>
      <td>ResNET18</td>
      <td>50.75</td>
      <td>33.51</td>
    </tr>
  </tbody>
</table>

### Direct Fine-tuning on Target Dataset Accuracy

Then in our next experiment, we fine-tuned our classifiers directly on target domain to get an upper bound of
accuracies on target domain for each classifier.

<table class="table table-bordered">
  <thead class="thead-dark">
    <tr>
      <th scope="col">Classifier</th>
      <th scope="col">Target Dataset 1 Accuracy (Used in Fine-tuning)</th>
      <th scope="col">Target Dataset 2 Accuracy (Unseen)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VGG16</td>
      <td>92.23</td>
      <td>42.75</td>
    </tr>
    <tr>
      <td>ResNET18</td>
      <td>96.47</td>
      <td>43.03</td>
    </tr>
  </tbody>
</table>

### Confusion Matrices for BaseLine Models

Confusion Matrices for baseline results. (a) VGG16 results on Target Dataset 1, (b) VGG16 results on Target Dataset 2, (c) ResNET18 results on Target Dataset 1, (d) ResNET18 results on Target Dataset 2

<img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/Confusion%20Matrices%20for%20baseline%20results.JPG" alt="Results" style="display: block;  margin-left: auto;  margin-right: auto;">

###  Confusion Matrices for Models fine-tuned with Target Dataset

Classifiers fine-tuned on target dataset directly. (a) VGG16 results on Target Dataset 1, (b) VGG16 results on Target Dataset 2, (c) ResNET18 results on Target Dataset 1, (d) ResNET18 results on Target Dataset 2

<img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/Classifiers%20fine-tuned%20on%20target%20dataset%20directly.JPG" alt="Results" style="display: block;  margin-left: auto;  margin-right: auto;">

### Baseline Trained Models

Baseline trained models can be found <a href="#" target="_blank">here</a>.

## WGAN

Our first approach to attempt domain adoptation problem was to use Wasserstein Generative Adversarial Networks. Our intention was to generate target like samples using WGAN and then fine tune VGG16 and ResNet18 with these newly generated images.But as below results depict, we were not able to generate high quality images so we had to abandoned this approach.
<figure class="image"><img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/WGAN%20Results.jpg" alt="System Diagram"  style="display: block;  margin-left: auto;  margin-right: auto;"><figcaption></figcaption></figure>

### Training Specifications of WGAN Models

<table class="table table-bordered">
  <thead class="thead-dark">
    <tr>
      <th scope="col">Specifications</th>
      <th scope="col">WGAN Arch. 1</th>
      <th scope="col">WGAN Arch. 2</th>
      <th scope="col">WGAN Arch. 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Model Type</td>
      <td>Linear</td>
      <td>Linear</td>
      <td>Convolutional</td>
    </tr>
    <tr>
      <td>Training
Epochs</td>
      <td>10K</td>
      <td>1K</td>
      <td>7.5K</td>
    </tr>
    <tr>
      <td>Training Time </td>
      <td>7 Days</td>
      <td>3 Days</td>
      <td>7 Days</td>
    </tr>
  </tbody>
</table>

### WGAN Trained Models 

WGAN Trained models are available <a href="#" target="_blank">here</a>.

## CycleGAN

### Models Fine-Tuned with CycleGAN Translated Images

Our Second approach was to translate source domain images into target domain images and then fine tune our models using these translated images.We trained seven different CycleGAN models, one for each emotion, to transalate source into target domain.Below are the reported accuracies for VGG16 & ResNet18 when fine-tuned with CycleGAN transalted images.

<table class="table table-bordered">
  <thead class="thead-dark">
    <tr>
      <th scope="col">Classifier</th>
      <th scope="col">Target Dataset 1
Accuracy (Used
in Fine-tuning)
</th>
      <th scope="col">Target Dataset 2
Accuracy (Unseen)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNET18</td>
      <td>48.13</td>
      <td>33.42</td>
    </tr>
  </tbody>
</table>

### CycleGAN Results

<figure class="image"><img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/CycleGAN%20translated%20images.jpg" alt="System Diagram"  style="display: block;  margin-left: auto;  margin-right: auto;"><figcaption></figcaption></figure>


### Confusion Matrices for Models Fine-Tuned with CycleGAN Translated Samples 

Classifiers fine-tuned on CycleGAN translated samples. (a) ResNET18 results on Target Dataset 1, (b) ResNET18 results on Target Dataset 2

<img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/CycleGAN%20Confusion%20Matrix.JPG" alt="Results" style="display: block;  margin-left: auto;  margin-right: auto;">

### CycleGAN Trained Models

CycleGAN trained models are available <a href="https://drive.google.com/drive/folders/1YQMcbfqQBPzH-AnjzQUshtOQNGcIlIF_?usp=sharing" target="_blank">here</a>.

## Feature Space Unsupervised Domain Adaptation
We retrained both the classifier with an additional domain classifier network in them. This domain classifier network helps in making the features used in classifier independent of any domain information.

<table class="table table-bordered">
  <thead class="thead-dark">
    <tr>
      <th scope="col">Classifier</th>
      <th scope="col">Target Dataset 1 Accuracy (Used in Fine-tuning)</th>
      <th scope="col">Target Dataset 2 Accuracy (Unseen)</th> 
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VGG16</td>
      <td>51.36</td>
      <td>37.37</td>
    </tr>
    <tr>
      <td>ResNET18</td>
      <td>46.72</td>
      <td>32.41</td>
    </tr>
  </tbody>
</table>

### Confusion Matrices for Models Trained Using Feature Space Domain Adaptation 

Classifiers trained using feature space domain adaptation approach. (a) VGG16 results on Target Dataset 1, (b) VGG16 results on Target Dataset 2t, (c) ResNET18 results on Target Dataset 1, (d) ResNET18 results on Target Dataset 2

<img src="https://raw.githubusercontent.com/adaptivefer/adaptivefer.github.io/master/assets/images/Classifiers%20trained%20using%20feature%20space%20domain%20adaptation%20approach.JPG" alt="Results" style="display: block;  margin-left: auto;  margin-right: auto;">


### Feature Space Domain Adaptation Models

Feature Space Unsupervised Domain Adaptation models are available <a href="#" target="_blank">here</a>.

## Downloads

### Models 
#### <a href="#" target="_blank"> Baseline Trained Models </a>
#### <a href="#" target="_blank"> WGAN Trained Models </a>
#### <a href="https://drive.google.com/drive/folders/1YQMcbfqQBPzH-AnjzQUshtOQNGcIlIF_?usp=sharing" target="_blank"> CycleGAN Trained Models </a>
#### <a href="#" target="_blank"> Feature Space Domain Adaptation Models </a>

### Dataset
#### <a href="https://drive.google.com/drive/folders/1_wnEIReyglKoNDoFy_v5IPsJMrBEcijV?usp=sharing" target="_blank"> Pakistani Dataset 1 </a>
#### <a href="https://drive.google.com/drive/folders/1STirPRvqy9e0Wg-blXqyq8LzygnJUSZu?usp=sharing" target="_blank"> Pakistani Dataset 2 </a>
#### <a href="https://drive.google.com/drive/folders/13u0iPDFO9M10qrY529XiunLabDFaNe3m?usp=sharing" target="_blank"> RAF_DB Western Dataset </a>


