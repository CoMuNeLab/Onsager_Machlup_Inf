30-Jul-2026   
   
Dear Dr Zambelli   
   
Your manuscript has now been peer reviewed and the reviews have been assessed by an Editor. The non-confidential reviewers’ comments and Editor comments are included at the end of this email and we would like to invite you to revise your manuscript to address them.    
   
As we only allow a limited number of revisions, we urge you to make every effort to fully address all of the comments at this stage. Your manuscript will be sent back to one or more of the original reviewers for assessment. If the original reviewers are not available, we may invite new reviewers. Please note that we cannot guarantee eventual acceptance of your manuscript at this stage.   
   
Please submit a copy of your revised paper within \<strong\>three weeks\</strong\>. \<strong\>If you are unable to meet this deadline, please let us know as soon as possible as you won’t be able to submit a revision if you miss the deadline. \</strong\>  
   
To submit your revision please \<a href="https://mc.manuscriptcentral.com/jrsi" target="\_blank"\>log in\</a\> and enter your Author Centre, where you will find your manuscript title listed under ’Manuscripts with Decisions’. Under ’Actions’ click on ’Create a Revision’. Your manuscript number has been updated to denote a revision.   
   
When submitting your revision please include the following: 

1\. A \<strong\>‘Response to Reviewers’ file\</strong\>: This should document, point by point, how you have responded to the reviewers’ and Editor comments and the adjustments you have made to the manuscript.  

2\. Your \<strong\>main manuscript\</strong\> should be submitted as text file (doc, tex). If you have used a tex file, please also upload a pdf of your main manuscript as well. We require a clean version of your manuscript (uploaded as main manuscript) and a version that clearly shows the edits made (for instance as tracked changes, bold text, or coloured highlights) and should be uploaded as 'Tracked changes main document’. 

3\. Your \<strong\>figures\</strong\> (PNG, TIFF, JPG, JPEG, and EPS) should be submitted as separate files and not included within the main manuscript file. 

4\. \<strong\>Electronic supplementary material (ESM)\</strong\>

5\. \<strong\>Ethics statement\</strong\>: If your study contains research on humans, please ensure that you include details of informed consent from participants and ethical approval from your local research ethics committee. If your study uses animals, please include details of any approval and licences given to carry out the study and include full details of how animal welfare standards were ensured. Field studies should be conducted in accordance with local legislation; please include details of the appropriate permission and licences that you obtained to carry out the field work.   
   
6\. Data accessibility and data citation: It is a condition of publication that \<strong\>data and code \</strong\>supporting your paper are made available in a repository or supplementary material for peer review. Authors must complete the ‘data accessibility’ section in the submission system. This should list the database and accession number for all data from the article that has been made publicly available. Please see our \<a href="https://royalsociety.org/journals/ethics-policies/data-sharing-mining/" target="\_blank"\>data policies\</a\> for more information.   
   
Information on language editing services can be found on our website \- \<a href="https://royalsociety.org/journals/authors/language-polishing/" target="\_blank"\>https://royalsociety.org/journals/authors/language-polishing/\</a\>. 

   
Thank you for submitting your manuscript to Journal of the Royal Society Interface. We look forward to receiving your revision. If you have any questions at all, please do not hesitate to get in touch.   
   
With best wishes   
   
Dr Maribel Hernández-Rosales  
Senior Editor    
interface@royalsociety.org 

Associate Editor  
Comments to Author(s):  
Both reviewers raised substantial questions about the quality of the manuscript, and robustness of claims made.

Referee(s)' Comments to Author(s):  
Referee: 1

Comments to the Author  
This manuscript reports a Bayesian ensemble inference framework for GRN inference based on the Onsager-Machlup path probability formalism, coupled with a node-wise block-coordinate Metropolis-Hastings sampler and an evidence-based model comparison scheme. While the theoretical grounding is notable, the manuscript has significant limitations in its applicability to real biological data, scalability, and empirical validation that must be addressed before publication.

Major concerns

1\. The GRN interaction weights in matrix A are restricted to {−1, 0, \+1}, ignoring heterogeneity in regulatory strength across gene pairs.  
2\. The method has serious scalability issues: the individual regulator inference step samples over all possible regulator combinations, the ensemble step scales exponentially with network size, and grid search over parameters is computationally expensive. The authors must provide a systematic scalability evaluation.  
3\. The method assumes that kinetic parameters across all genes are identical, which is an overly simplified assumption. It is known that gene-specific rates can vary by orders of magnitude.  
4\. The prior choices of sparsity and enrichment of positive feedback loops are presented as biologically motivated but not rigorously evaluated or benchmarked. Performance will depend heavily on how well these priors match the ground-truth topology.  
5\. The empirical evaluation rests on a single synthetic test case with no real biological data and no comparison to existing methods.

Minor concerns:

1\. There are formatting issues in some section titles, with multiple sections labeled "A." (e.g., lines 211 and 268).  
2\. The decision to treat kinetic parameters as fixed is not made clear until line 212 and should be stated earlier in the Methods.  
3\. The claim at line 217 that GRN connectivity is sensitive to changes in other parameters is not adequately supported (Figure 2C gives some indirect hints but still cannot serve as direct support).  
4\. Insufficient information is provided on the simulated cases, including model parameters and availability of the simulated data.

Referee: 2

Comments to the Author  
This paper proposes a methodology to infer the structure of a network from different snapshots of a non-linear dynamic on top of it. Although the title suggests its use for Gene Regulatory Networks (GRN), in fact, GRN play almost no role in the paper; it is just used to justify the chosen dynamics and details on the expected structure (priors, and type of network). The methodology is interesting, and the results on a small-scale example confirm the ability of the method to discover most of the structure. However, the manuscript has many problems that must be solved before considering it for publication:

1\) As commented above, the paper does not deal with real GRN, as could be interpreted from the title. Please change the title to one that really reflects the content of the paper, and modify the rest of the manuscript accordingly.

2\) Please explain what an SDE is in its first appearance (not on page 4).

3\) You use at least three different notations for the same mathematical object, which is quite confusing: A \= W \= Theta. And maybe also w.

4\) When matrix A first appears, it is not clear if it can be symmetric or not. Later, only the “asymmetric” option seems possible, since otherwise the decomposition in Eq. (5) would not be possible. This is just an example, but in general, the paper is written in such a way that it unnecessarily forces the reader to try to discover things that simply should have been explicitly stated in the paper.

5\) In Eq. (6), the introduction of beta in this way breaks the interpretation of this equation as a consequence of Bayes' rule. I would understand that pi(Theta) could be written as pi(Theta, beta). Anyway, this parameter beta is introduced here and disappears for the rest of the paper. Why? How do you set its value? Which values have you used in the experimental part? Is this really needed?

6\) In Eq. (8) a P(Theta) appears in the place for the prior pi(Theta), or pi(Theta, beta). I’m totally confused; once again, new notations appear without explanation.

7\) In Section IV.A, you try to separate the inference of the dynamical parameters from the structure of the interactions. Instead of adding the dynamical parameters into the inference process, you just fix them (using a grid search), apply the inference method to the structure of the network, and then use the found likelihood to score the different values of the dynamical parameters. This seems a reasonable approach, but some questions arise: Have you tried both approaches (grid search and joint inference) to be sure that the grid search is the right solution? If so, the results should be reported as evidence for the benefits of this procedure. Additionally, in Figure 2c, you show a grid with a huge number of values for only two of the parameters, but there are more parameters. Have you fixed them to the real values? Or is the grid for the four parameters simultaneously? Finally, how do you fix the minimum and maximum values for the grid search? All the parameters are unbounded, so you may need to scan multiple orders of magnitude.

8\) In Section VII.A (the first Materials and Methods), the content is impossible to understand: no references, too many undefined symbols (what are F\_i, gamma\_i, omega\_ij, etc.?), unjustified hypotheses (e.g., why F\_i is Lipschitz-continuous), undetailed deductions, etc. And most importantly: what are you trying to prove?

9\) In the experiments section, you say “The specific choice of the parameters and the network topology is reported in the Materials and Methods section.” It is not reported. In fact, all the experiments are impossible to reproduce, since there are few details on them. Even the results are difficult to understand. What does the activity distribution in Figure 3a mean? I guess you refer to the values of x\_i, which were initially named states, not activities. In any case, they evolve in time, so do you plot the distribution of temporal values?

10\) In the definition of the prior for the feedback loops, it seems awkward to define N\_neg in a way that yields negative values, since what you want is to count the number of negative feedback loops. It seems more reasonable to add a minus sign before the lambda in the exponential and reverse the terms in the calculation of N\_neg. Moreover, in the current version, P(G) seems to be able to take values greater than one, in contradiction to the fact of representing a probability. Finally, once again, P(G) is a new unexplained notation, so add G to the set of different names for matrix A \= W \= w \= Theta \= G.

11\) Given that this work is basically to present a methodology to calculate something, I would expect that the code would be available as open source.

In summary, I’m surprised by the lack of previous revision of the paper before submission; this is not serious and should not happen. Even if the methodology makes sense and seems useful, the authors should have made a much better effort in its preparation.