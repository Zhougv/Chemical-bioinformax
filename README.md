# Chemical-bioinformax
To solve the problem of lack of label data and inadequate integration of multiple data, we develop a new multi-dimensional information fusion self-supervised molecular representation algorithm. We enhance the pre-training process by using additional intermolecular data to pretrain a model. Our approach is called Chemical-bioinformax. Through the pre-training of node-level and graph-level tasks, better performance is obtained on graph-level and edge-level tasks, and sota performance has been obtained in the mutual information molecular pre-training method.

![image](https://github.com/Zhougv/Chemical-bioinformax/assets/164281953/32bd28bc-e02c-408e-a986-4c501786c26a)
Figure 1 shows the overall flow of pre-training.
![image](https://github.com/Zhougv/Chemical-bioinformax/assets/164281953/32371ee0-84fd-46d4-a6a4-93837019f8f4)
Figure 2 shows the overall flow of fine-tuning.
The sample data and representation data we used are from chemical checker (2020NBT). Chemical checker is a multi-omics small molecule representation library. A 2021 NC paper continued this work by making it possible to calculate each small molecule's chemical checker representation. If you want to use the original chemical checker data, you can visit https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137202. If you want to generate chemical checker data for your small molecules, you can pip install signaturizer.
The code will be available when our article is published
