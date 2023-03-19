# Who-wins-TF-IDF-vs-LDA-on-Topic-Modelling
This NLP project contains topic modelling with Term Frequency-Inverse Document Frequency (TF-IDF) and Latent Dirichlet Allocation techniques on company descriptions dataset.

<p align="center" width="100%">
    <img width="70%" src="figures\topic_modelling.png">
</p>

<p align="center"width="10%">
        <em>Source: Blei, D. (2012). Probabilistic Topic Models. Commun. ACM, 55(4), 77–84.</em>
</p>    

### Project Goal
The goal of this project is to develop aforementioned techniques on topic modelling ad compare their outcomes.
The company descriptions dataset consists of company name, descriptions, foundation year, relevancy label.

### Respiratory Structure
- `data`: A directory containing the dataset.
- `model`: A directory including trained LDA model.
- `results`: A directory containing results of TF-IDF and LDA methods on the query company.
- `notebooks`: A directory containing EDA and LDA visualization notebooks.


### Script
1. Install `virtualenv`: ```pip install virtualenv```

2. Create virtual environment `myenv`: ```virtualenv myenv```

3. Activate the environment: ```myenv\Scripts\activate```

4. Install the required packages: ```pip install -r requirements.txt```

5. Conduct exploratory data analysis: notebooks >> ```EDA.ipynb```

5. Run topic modelling experiments:

    ```python main.py --experiment LDA --co_name "Vahanalytics" --top_k 5 --num_topics 4 --chunk 2000 --alpha "auto" --iterations 1000 --passes 10 ```

7. Analyze LDA results with visualizations: exploratory data analysis: notebooks >> ```LDA_visualization.ipynb```


### LDA Visualization

<p align="center" width="100%">
    <img width="100%" src="figures\lda_vis2.png">
</p>

### Examples

**Query Company:** Vahanalytics
**Company Description:** Vahanalytics aims to create better drivers and safer roads by using cutting edge big data and machine learning techniques.

#### TF-IDF Results
Top 1: BISAF

BISAF is a technological company for the construction industry. We specialise in cutting edge solutions that make building easier, safer and environmentally friendly.

Top 2:  GeoSpock

GeoSpock brings together their expertise of big data engineering to unlock the hidden value of data silos in your organization. Their solution enables you to manage extreme amounts of data at speed enabling your organization to react to key insights in a timely manner for future business success. The technology enables a range of capabilities from data analytics, visualization of spatial data, cutting edge data indexing, custom querying of data sets, and data intelligence.

Top 3:Axenda

Axenda is a cloud-based software platform for construction management industry. The software platform is used by constructors and architects to manage day-to-day tasks and grow their businesses. The company's patent-pending algorithm uses machine learning to estimate materials & resources. It aims to predict project's estimates & completion deadlines. In addition, the platform also translates the data into 3D virtual models which give visual feedback of project's progress to clients.


#### LDA Results

Top 1: CMA Supply

CMA Supply is a regional distributor of concrete and masonry products and accessories to professional contractors in the residential, commercial, and industrial end-markets.  The Company was founded by William Updike in June of 1978 in Indianapolis, Indiana and has grown to six distribution facilities and two rebar and fabrication facilities across Indiana, Kentucky, and Ohio. The Company has a premier reputation for service and product selection within the region.

Top 2:  Tiger Calcium

Tiger Calcium is a  manufacture, supply, transport and apply premium calcium chloride products. Tiger Calcium is a producers of calcium chloride products in North America, drawing from the largest known reserve of naturally occurring calcium chloride. Handling all areas of production while also managing our own dedicated transportation fleet enables us to produce a consistently superior product with an ensured supply throughout the year.

Top 3:Pan United Corporation Ltd

Pan-United Corporation Ltd (PanU) is an Asian multinational corporation leading in specialised concrete solutions and a global leader in concrete technologies. PanU harnesses cutting-edge technology to develop high-performance, sustainable concrete products. It is Singapore’s largest supplier of ready-mixed concrete and cement. Supported by a total workforce of more than 1,200 people, PanU thrives on innovation, operational excellence and long-termism.
