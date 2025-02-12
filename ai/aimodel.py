import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


# Define a dataset class
class ScientificDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(
            self.labels[idx], dtype=torch.float
        )  # Adjust for regression
        return item


# Mock data
texts = [
    """Chronic illnesses are a major threat to global population health through the lifespan into older age. Despite world-wide public health goals, there has been a steady increase in chronic and non-communicable diseases (e.g., cancer, cardiovascular and metabolic disorders) and strong growth in mental health disorders. In 2010, 67% of deaths worldwide were due to chronic diseases and this increased to 74% in 2019, with accelerated growth in the COVID-19 era and its aftermath. Aging and wellbeing across the lifespan are positively impacted by the presence of effective prevention and management of chronic illness that can enhance population health. This paper provides a short overview of the journey to this current situation followed by discussion of how we may better address what the World Health Organization has termed the “tsunami of chronic diseases.” In this paper we advocate for the development, validation, and subsequent deployment of integrated: 1. Polygenic and multifactorial risk prediction tools to screen for those at future risk of chronic disease and those with undiagnosed chronic disease. 2. Advanced preventive, behavior change and chronic disease management to maximize population health and wellbeing. 3. Digital health systems to support greater efficiencies in population-scale health prevention and intervention programs. It is argued that each of these actions individually has an emerging evidence base. However, there has been limited research to date concerning the combined population-level health effects of their integration. We outline the conceptual framework within which we are planning and currently conducting studies to investigate the effects of their integration.
Introduction

The world is slowly emerging from one of its most challenging periods in modern human history with the COVID-19 pandemic and its aftermath. Before the COVID pandemic the focus was on Chronic/Non-Communicable Diseases (NCDs). The WHO President (1) then warned about the growth in chronic disease burden. The significance of NCDs was recognized in the United Nations 2030 Agenda for Sustainable Development, which set targets to, “reduce by one third premature mortality from noncommunicable diseases through prevention and treatment, and promote mental health and well-being” (2). Unfortunately, it is now clear that COVID-19 has contributed to major upticks in underlying and consequential chronic illnesses and diseases from an already high base (3). We now seem to be in an even worse position than before. We have a syndemic driven by the existing chronic disease pandemic overlaid by the newer COVID-19 pandemic (4). A key challenge in the syndemic is to provide adequately resourced and well-trained public and clinical health workforces (5–7).
The burden of chronic diseases and illness in the human population

The burden of chronic illness globally has increased over time, accounting for the major part of global disease burden (8, 9). The epidemiology of chronic disease burden varies from country to country but most countries whether developed or developing have high chronic disease burden. The Lancet global burden studies chronicle the burden of 369 diseases and injuries in 204 countries and territories. They show that chronic conditions caused 74% of all deaths worldwide in 2019, rising from 67% of deaths in 2010. The mortality data reflect high prevalence’s of chronic conditions across populations. For example, in Australia, the Australian Institute of Health and Welfare (10), has noted 47% of Australians have at least one chronic disease with 20% having 2 or more. 51% of hospitalizations involve chronic disease, 90% of deaths and disease burden is borne disproportionately by adults of lower SES and those living in remote areas. These results are consistent with those globally and in many countries. The USA NIH (11) has noted “currently, some 50% of the US population has a chronic disease, creating an epidemic, and 86% of healthcare costs are attributable to chronic disease.” In the United Kingdom a similar epidemiological pattern is evident with close links between traditional physical chronic diseases and associated mental health disorders (12). These studies show increased prevalence across the lifespan threatening healthy aging.
The contribution of mental health disorders, including addictions, to population disease burden

Mental health disorders including addictions are now an increasing challenge facing humanity (13, 14) with 20 percent of global disease burden. One might be forgiven for labeling recent decades as the “Age of Addiction” (15) with the addiction “traditionals” of alcohol, cannabis, and other substances and new synthesized agents overlaid by newly recognized behavioral addictions (16) (see the WHO ICD (17) and AMA DSM) (18). Of the behavioral addictions, the first was Gambling Disorder (GD) – which continues to be neglected in terms of research and development of new interventions. GD is linked to high levels of mental and physical health comorbidities, health economic costs, homelessness, and suicidality (19, 20). In addition to addictions, there is strong growth in the population prevalence’s of mood disorders and anxiety (21). Interpreting the true growth in mental health disorders has some nuances. Recognition of the importance of mental health disorders (12, 22) has also led to increased prominence in national health surveys and public health epidemiological studies. These changes in population health study content make it difficult to assess the true extent of the underlying growth in the population prevalence of such disorders. Another problem is that the most pressing mental health conditions are seldom appropriately measured in public health epidemiological studies – especially GD. These issues are examples of “what gets measured gets managed” and contrariwise. There are many instances where inclusion of measures has improved health services and policy making (23). For example, mental health measures are now an increasingly prominent component of clinical datasets and such conditions are receiving greater clinical effort and funding. The same is true for chronic diseases in general which are now centrally located in health policy and service design in many countries.

The evidence base provided by disease burden and epidemiological studies strongly reinforces the WHO’s alarm. What therefore is to be done? How can the global chronic diseases/long-term conditions pandemic be better addressed? Below we recommend an integrated population-level approach involving (1) large-scale measurement of polygenic and multifactorial risk factors in order to develop and rigorously validate clinically useful prediction tools and algorithms; (2) early and sustained, effective management of chronic diseases using advanced behavior change interventions and (3) digital health approaches to improve the efficiency and reach of interventions and health services at a wider population level.
Development, validation, and deployment of risk prediction tools using polygenic and multifactorial risk data

Early identification of people at risk of chronic illness and early intervention are key to reducing population chronic disease burden. Unfortunately, this obvious game changer is infrequently implemented in many public health regimens. We now have much better technology available to develop and validate evidence-based risk prediction tools and algorithms, and to demonstrate their value by leveraging digital tools that can be embedded within at-scale screening and treatment programs. We already have the technology to develop and validate useful risk prediction tools and algorithms but we contend that we are not yet systematically conducting such research and implementing resulting tools at the required pace in large-scale studies as outlined in the recent International Common Disease Alliance Polygenic Risk Score Task Force report (24). The first contact with many people with chronic disease risk is after they have already developed it (25, 26). For mental health disorders, delays in presentation and intervention can be particularly long, adding to the burden of disease. Obsessive-compulsive disorder (OCD) is one of the top ten leading causes of disability in the developed world and has a typical duration of untreated illness of 10 years (27). While less well studied, a similar duration of untreated illness has been reported for gambling disorder – around 9 years in affected individuals presenting for treatment (28).

Polygenic and multifactorial risk prediction can play a major role in delivering early warning of impending chronic diseases (29) including traditional chronic diseases such as cardiovascular disorders (30), metabolic disorders (31), and cancers (32). While more research is certainly needed, initial data suggest some promise for mental health disorders (33) including gambling disorder and newer concepts such as gaming disorder (34, 35). Wider implementation of such strategies has the potential to drive down the costs of what are now mature and proven technologies, but already they are affordable. The costs associated with inaction with chronic diseases are substantial (36). The potential cost reductions in health care costs and the net benefits of prevention, early detection and intervention are well established in principle. It is our view that humanity cannot afford to further delay polygenic and multifactorial risk prediction, early diagnosis and intervention.

Beyond the need for data collection and linked rigorous validation, to evaluate the potential value of incorporating polygenic and multifactorial risk prediction into routine practice, the global public health and health care workforces need to be trained to use these tools, to effectively communicate the meaning of risk and risk management to the community. Such training is relatively common for some conditions but neglected in other conditions. In the United Kingdom, there has been recent work to promote the wider use of genomics in General Practice (37, 38).
The prevention and management of chronic diseases

Lifestyle risk factors make a major contribution to the chronic disease burden over the life course. Many of the same risk factors contribute to multiple chronic diseases. Given the high contribution of these behavioral risk factors to multiple chronic diseases, and, that these diseases comprise such a high proportion of total disease burden, it is obvious that public health and clinical workforces need strong chronic disease program prevention skills. Inclusion of such matters in the medical and clinical health curriculums is an essential and welcome innovation to contemporary chronic disease management (39–41).

It is important to understand that treatment for the various chronic diseases must follow recognized evidence based clinical and population health prevention guidelines. Study of these guidelines show that there is a high degree of commonality in the risk factors that contribute to chronic diseases. Advanced behavior changes skills facilitate more effective prevention and management of chronic disease.

Some of this work on expanding training across health systems has been conducted by members of the authorship team. The training has been directed at public health practitioners and clinical workers in public health programs. The Happy Life Club originated in Australia and was then translated to China in various major cities and provinces where it has grown strongly (42–44). Initial economic evaluation (45) showed that incremental benefit for each patient corresponded to $AUD 16,000 over an 18-month period. The 2020 frontiers special issue devoted to chronic disease and aging (46, 47) built on other work concerning Chronic Disease Management (CDM) programs (48). The Club program trains public health practitioners and clinicians to prevent and manage chronic conditions using Motivational Interviewing (MI) principles (49).

The Happy Life Club coach training program has been (50–52) studied as the subject of evaluations and the program was the subject of a large Randomized Controlled Trial (53) and is a World Bank recommended intervention (54). While MI is a central part of the program, rigorous outcome measurement using validated tools and patient-centered care principles are also key components. These techniques are more broadly applicable across a range of chronic diseases including mental health and addiction disorders (55–57) in controlled trials. It would seem sensible to conduct clinician and public health training so that MI techniques can be more widely applied in public health settings with the aim of further reducing the impact and burden of chronic mental health symptoms.
Digital health platforms: improving population reach and program efficiency

Digital health platforms hold the potential to facilitate Chronic Disease Management across the lifespan at scale in health systems. A large-scale review of digital health platforms was conducted by WHO and the Cochrane Collaboration to develop the Digital Interventions for Health System Strengthening guideline (58). Eleven new Cochrane reviews were included in the guideline. The guideline highlighted different applications of digital health including conventional public health programs, prevention, clinical delivery and back-office programs. It also highlighted the need for the collection of more evidence for these platforms. A 2022 European review (59) investigating the cost-effectiveness of digital health interventions concluded that the evidence was not yet sufficient to return a positive or negative conclusion. We are committed to providing effective digital health support to public health and clinical workers for chronic disease prevention and intervention programs. The goal is to address the knowledge as to how such tools may be best applied (60–63).

Of course, as noted, it is important to be mindful that specific tools are needed for specific purposes and generalizing across all tools is of limited value. Examples of recent digital tool development work in the field of alcohol use include a tool that can estimate weekly alcohol intake based on responses to the extended AUDIT questionnaire; and a web-app brief intervention to raise awareness about the impact of alcohol on breast cancer in a breast cancer clinic setting (potentially modifiable risk factors account for around 25% of breast cancer cases (64, 65)). Another example is that in work led by UK members of our group, we have developed and are piloting a digital tool for NHS gambling treatment services, which collects validated assessment and outcome data from affected individuals and generates readily interpretable summary reports, which are then discussed by the clinician and their patient. This approach could have potential advantages in terms of streamlining the clinical assessment process, fostering early and sustained patient engagement, and improving quality and volume of research data to improve care pathways.

Overall, digital tools are at different levels of development and only some could be deemed sufficiently validated for current widespread use. However, we feel these examples highlight the potential utility of such tools for public health prevention and interventions, in the management and mitigation of chronic diseases. Smart technologies are now available with much-improved access. Recent data shows that access to the Internet of Things will grow from 15 billion to 30 billion devices in the next 7 years (66). Eight billion of these devices will be smartphone connections (67). It is well understood that the availability of smart devices has been less among economically disadvantaged groups. However, this is not cause to deny the obvious advantages of using smart device technologies in global prevention and management of chronic disease of the majority of people.
The importance of rigorous evaluation and validation of approaches to chronic disease management and prevention

As previously stated, there is significant evidence for the efficacy of polygenic and multifactorial risk screening, strong evidence for the efficacy of advanced behavior change principles and a developing evidence base for the efficacy of digital health platforms for some conditions and populations (68–70). Rigorous evaluation of the impact of these approaches in combination and their efficacy across varied populations and chronic diseases and disorders is required. The systematic reviews that have been done of the cost-effectiveness of programs to address chronic diseases show significant advantages (71) but many studies do not include adequate economic analysis or active comparator/control conditions.

The conduct of health economics modeling is a key plank of the valuation of chronic disease management programs. Using tools such as the EuroQoL suite and the use of look up tables developed from major studies credible Disability Adjusted Life Years (DALY) and Quality Adjusted Life Years (QALY) estimations can be constructed from the data (72).

There is the technical issue of what methods of economic evaluation ought to be used to assess the costs and benefits of the study outcomes based on the collected data (73). Many studies use an ICER (incremental cost-effectiveness ratio) or an INMB (incremental net monetary benefit) of the intervention compared with usual programs. Both ICER and INMB methods have limitations, but ICER is currently more widely used. INMB has some appealing aspects. A key one is that it is couched in terms of dollar values whereby costs and benefits are expressed in the same dollar value units. This provides not only ease of interpretation but also the ability to use it to enable direct comparison across different programs. All studies in the program will conform to the design principles outlined in the Bias in Economic Evaluation (ECOBIAS) standards (74) and Consolidated Health Economic Evaluating Reporting Standards (CHEERS) (75) to guide the reporting of study outcomes and rigorous design.
Summary of the suggested multidisciplinary population chronic disease prevention and management approach INTEGRATE

The following chart outlines the logic of the INTEGRATE program approach that aims to enhance improved health outcomes across the lifespan through, prevention, earlier detection and more effective intervention. It is intended that the INTEGRATE model will be applied to a range of chronic physical and mental health conditions at a population level, reflecting the high degree of multi-morbidity identified in population and clinical studies. The INTEGRATE model does not replace guidelines-driven programs and interventions. It provides an organizing framework for important strategic disease risk prediction data based on genomics science and multi-factorial risk assessment, supports public health workers and clinicians using their disciplinary evidence-based prevention/ treatment guidelines to deal with a range of chronic illnesses by enhancing their behavior change skills and assists with the cost-effective delivery of treatment by utilizing advanced digital health platform capabilities. The public health and clinical workers are augmented by powerful support tools (Figure 1).
FIGURE 1
www.frontiersin.org

Figure 1. INTEGRATE program logic map.

The INTEGRATE model seeks to combine polygenic genomic and diagnostic testing and history data for target chronic illnesses to identify sub-populations that are low risk, at risk and with diagnosed and undiagnosed conditions. Those with high risk but no diagnosed condition are referred for preventive actions to lower their risk. Those with diagnosed conditions are referred into treatment programs to improve health and wellbeing. All cases in the preventive and treatment programs may be tracked to assess their ongoing health status and wellbeing. These program actions are powered by behavior change science and prevention and treatment guidelines pertinent to their chronic illnesses assisted by digital platform technologies. It is intended that cases and at-risk community members will be detected and enter earlier preventive and treatment programs. Health economics, participant experience and health outcomes studies will be used to evaluate program effectiveness and efficacy. It is intended that health care costs will be lowered, risk reduced, and outcomes improved by the application of this model through its integration into public health and clinical programs targeting chronic diseases.

We have now set ourselves the task of evaluating the efficiency and effectiveness of the INTEGRATE approach we have described for integrated prevention and effective interventions across a range of chronic diseases. We believe that this approach addresses several key problems. Although there have been many exhortations of the virtues of prevention and early invention among at-risk populations, the genomic technologies that practically and expediently this approach are recent, but in some cases are now sufficiently developed to trial and use at scale. However, we must take an evidence-based and skeptical approach using health economics and rigorous clinical efficacy evidence including appropriate control conditions (where feasible). Behavior change science has been with us for several decades, but its power has not been fully implemented due to lack of contemporary training throughout the public health and clinical workforces. This science does not replace, for example, pharmacological and other interventions, it complements and augments them.

We feel these promising technologies are within our grasp and now we have the duty of evaluating and implementing them in an effective integrated way to advantage the targeted populations within our communities. This integrated approach has considerable promise for promoting population health, healthy aging and reducing the current burdens of health care. We also have a commitment to not artificially separate “physical” and “mental” health conditions in a context where they are so demonstrably interdependent.
Data availability statement

The original contributions presented in the study are included in the article/supplementary material, further inquiries can be directed to the corresponding author.
Author contributions

ST and CB conceptualized the paper. ST drafted the paper and had final editorial approval. CB assisted with the drafting of the paper and provided editorial input. FC contributed to the genomic sections of the paper. BK contributed to the digital health sections of the paper. MO provided a public health perspective to the drafting of the paper. HB-J and SC assisted with drafting of the mental health sections of the paper. All authors provided editorial comments in addition to their substantive contributions.
Acknowledgments

The authors thank Liam Thomas for his contribution as a Research Officer to this work, assisting with literature searches and summaries, reference management and document handling.
Conflict of interest

SRC’s research is funded by the NHS and was previously funded by Wellcome (an independent charity). SRC receives a stipend for editorial work at Elsevier journals (Comprehensive Psychiatry, and Neuroscience & Biobehavioral Reviews). HB-J is the Director of the UK National Problem Gambling Clinic and the UK National Centre for Gaming Disorders which are now fully funded by the NHS. These clinics have previously received funding from NHS England, Central and North West London NHS Trust, and GambleAware. HB-J is Vice President of the Royal Society of Medicine and sits on several national and international Boards. HB-J has been on research teams funded by the Medical Research Council, the Wellcome Trust, and the Wolfson Family Trust.

The remaining authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.""",
    """Introduction

As the United States population ages, the adult population with chronic diseases is expected to increase. Exploring credible, evidence-based projections of the future burden of chronic diseases is fundamental to understanding the likely impact of established and emerging interventions on the incidence and prevalence of chronic disease. Projections of chronic disease often involve cross-sectional data that fails to account for the transition of individuals across different health states. Thus, this research aims to address this gap by projecting the number of adult Americans with chronic disease based on empirically estimated age, gender, and race-specific transition rates across predetermined health states.
Methods

We developed a multi-state population model that disaggregates the adult population in the United States into three health states, i.e., (a) healthy, (b) one chronic condition, and (c) multimorbidity. Data from the 1998 to 2018 Health and Retirement Study was used to estimate age, gender, and race-specific transition rates across the three health states, as input to the multi-state population model to project future chronic disease burden.
Results

The number of people in the United States aged 50 years and older will increase by 61.11% from 137.25 million in 2020 to 221.13 million in 2050. Of the population 50 years and older, the number with at least one chronic disease is estimated to increase by 99.5% from 71.522 million in 2020 to 142.66 million by 2050. At the same time, those with multimorbidity are projected to increase 91.16% from 7.8304 million in 2020 to 14.968 million in 2050. By race by 2050, 64.6% of non-Hispanic whites will likely have one or more chronic conditions, while for non-Hispanic black, 61.47%, and Hispanic and other races 64.5%.
Conclusion

The evidence-based projections provide the foundation for policymakers to explore the impact of interventions on targeted population groups and plan for the health workforce required to provide adequate care for current and future individuals with chronic diseases.

Keywords: chronic disease, adult population, multi-state population projection, United States of America, projections
What is already known about this topic?

    - Adults population in the United States with chronic diseases is expected to increase.

What is added by this research?

    - An evidence-based age, gender, and race-specific projections of the burden of chronic diseases show that the majority of the adult population 50 years and older, across all races, will have at least one chronic disease by 2050, with the majority between the ages of 60 to 79 years.

What are the implications for public health?

    - The importance of prioritizing the promotion of access to high-quality primary care to provide whole-person care that ensures prevention and management of chronic disease care and addresses evidence-based social determinants of health that increase the risk of developing chronic diseases.

Introduction

According to the US Centers for Disease Control and Prevention, in 2019, 54.1 million US adults were 65 years or older, representing 16% of the population. By 2040, it is estimated that the number of older adults 65 years and older is expected to reach 80.8 million and 94.7 million by 2060, representing 25% of the US population. An aging population is characterized by the co-occurrence of more than one chronic condition, which is referred to as multimorbidity (1–4).

A meta-analysis of the prevalence of multimorbidity in high, low, and middle-income countries found an overall pooled prevalence of 33.1% (30.0–36.3). There was a considerable difference in the pooled estimates between high-income countries and low and middle-income countries, with prevalence ranging between 37.9 (32.5–43.5) and 29.7% (26.4–33.0), respectively (5). In the United States, data from the 2018 National Health Interview Survey (NHIS) indicates that 27.2% of US adults had multiple chronic conditions. While multimorbidity is not new, there is greater recognition of its impact and the importance of improving outcomes for individuals affected. Multimorbidity is associated with increased mortality (6), reduced quality of life, and functional status (2, 7, 8), increased health services use (3, 9), and higher cost of care.

To better understand the future chronic disease burden, as well as explore the effectiveness of various interventions on the incidence and prevalence of chronic disease, including quality of life outcomes for people with chronic disease, requires an evidence-based and credible forecast of the current and a future number of American adults with chronic disease. Projections of chronic disease often involve cross-sectional data that fails to account for the transition of individuals across different health states. Thus, this research aims to address this gap by projecting the number of adult Americans with chronic disease based on empirically estimated age, gender, and race-specific transition rates across predetermined health states. The evidence-based projections from this research could help healthcare providers to implement interventions for targeted population groups to prevent and or manage their chronic disease and plan for the health workforce required to provide adequate care for current and future individuals with chronic diseases to achieve the quadruple aim of healthcare, i.e., improve population health, reduce cost, and increase patients' and providers satisfaction (10, 11).
Methods
Model design

To project the number of Americans 50 years and older with one or more chronic conditions, we developed and validated a dynamic multi-state population model (12–15) to simulate the population of the United States and track their transition to and from three health states. The health states are (a) healthy (adults with no chronic condition), (b) one chronic condition (adult with any one of the nine chronic conditions indicated in the Health and Retirement Survey), and (c) multimorbidity (adults with at least two chronic conditions indicated in the Health and Retirement Survey). For each health state, adult individuals were further divided into a three-dimensional vector: age (from age 50–100 and older), gender (male and female), and race (non-Hispanic white, non-Hispanic black, Hispanic, and other races). To ensure consistency and validation of the model output, an additional state that accounts for the population below 50 years was included to ensure that individuals aged 50 transitions to the adult population's health states. To ensure a consistent aging process, the population aged 50 years and younger was subdivided by age (age 0–age 49). The number of people below age 50 increases by births and net migration (estimated by calibration) and decreases by deaths and becoming age 50. Births were estimated using race-specific fertility rates from the National Vital Statistics report and the fecund female population age 15–49, while life tables informed deaths (16). At the end of each year, the surviving population in each age cohort flows to the subsequent cohort, except the final age cohort, age 100 and older. Transition across health states was determined by 1-year age-gender-race specific transition rates.
Health states

The chronic conditions in the Health and Retirement Survey record self-reported lifetime histories of a modest number of illnesses and conditions that are very important to older persons and account for much of the morbidity and mortality among older persons in western societies. The conditions consist of: (a) hypertension, (b) diabetes mellitus, (c) cancer (various types at all bodily sites, except minor skin cancers), (d) chronic lung diseases (often including emphysema, but not asthma), (e) coronary health disease, (f) congestive health failure, (g) stroke (cerebrovascular disease), (h) arthritis (a collection of heterogeneous diseases and Musculoskeletal pain syndromes), and (i) psychiatric problem (in general, not further defined or categorized, except major depressive, depressive symptoms, and dementia). Adult individuals who reported no presence of any chronic conditions were classified as healthy; those who reported only one of the chronic conditions were classified as adults with a single chronic condition, whereas those with at least two or more chronic conditions were classified as adults with multimorbidity.
Model assumption

Constant age-gender-race-specific mortality rates were used for the population 50 years and younger. For the adult population, the 1000 bootstrap estimates for all the transitions across the health states accounted for future improvement or deterioration. A race-specific fertility rate was used, while we assumed a constant fertility rate from 2018 over the simulation time. This assumption was deemed appropriate because a fertility rate change will not impact the adult population by 2060. The net migration rate, estimated via calibration, was assumed to be constant over the simulation time.
Estimation of transition rates

The 1998 to 2018 Health and Retirement Study data (17) was used to estimate the transition rates across health states. The Health and Retirement Study is a longitudinal panel study that surveys a representative sample of more than 26,000 Americans over the age of 50 every two years. The study explores changes in labor force participation and the health transitions that individuals undergo toward the end of their work lives. Since its inception, the study has collected information about income, work, assets, pension plans, health insurance, disability, physical health and functioning, cognitive impairment, and healthcare expenditure.

The input data to the transition rate estimate is in an extended format, and each observation occupies one line of data. Each line of observation includes the individual's age and values of covariates in the model. Everyone has multiple lines of observation. Since the Health and Retirement Study is not an annual survey, we fill in gaps with pseudo-data representing successive years to obtain annual transition probabilities (18). If starting and ending states of an interval are the same, the filled-in data assume that states. If an interval's starting and ending states differ, the filled-in data assume one transition at a random time. Multinomial logistic regressions are then fitted to estimate the probability of transitioning from a starting health state to one of ending health states (including death). Multinomial logistic regression models estimate age, gender, and race-specific transition rates.
Model validation

The model structure has been validated and used for several publications on similar chronic conditions in other countries (13–15). Thus, the structure of the model has been presented to researchers familiar with chronic disease care in several countries to verify the conceptual framework of the model and its assumptions regarding causal relationships as indicated in the literature cited (19–21). The model structure is grounded in evidence of how individuals transition from a healthy state to a single and multiple chronic conditions over their lifetime. To ensure that the model output is consistent with available data, selected simulated outcomes were compared with available data. The results suggest that the simulated model outputs compare favorably with the available data, demonstrating that the model performs credibly.
Results
Transition rates by age, gender, and race

Figure 1 shows the age, gender, and race-specific transition rates across the three health states and death. For both gender and race, the progression to worse health status (healthy to one chronic condition, healthy to multimorbidity, and from one chronic condition to multimorbidity) increases with age, except for the progression from healthy to one chronic condition where the transition rates begin to decrease significantly from age 90. Also, mortality across all the health states increases with age. On the contrary, for both gender and race, the regression to a better health status from multimorbidity to one chronic condition decreases with age.
Figure 1.

Figure 1
Open in a new tab

Transition rates across health states. Mwhite: is male non-Hispanic white; Mblack: is male non-Hispanic black; Mhispanic: is male Hispanic; and Mother: is male other races; while Fwhite: is female non-Hispanic white; Fblack: is female non-Hispanic black; Fhispanic: is female Hispanic; and Fothers: is female other races.

For gender differences, males have a higher rate of progression to a worse health state compared to females, while regression to a better health state was better for females than males. Also, females are more likely to maintain their health status than males. In the case of mortality, males have higher death rates than females. For race differences, non-Hispanic White had a higher transition rate while non-Hispanic Black had the lowest transition rate from a healthy to one chronic condition. For individuals transitioning from healthy to multimorbidity, Hispanics had a higher transition rate for all races, while non-Hispanic Whites had the lowest transition rate. Likewise, Hispanics had the highest transition rates for individuals transitioning from one chronic condition to multimorbidity, while other races had the lowest transition rates. Hispanics had the highest transition rates for regression from multimorbidity to one chronic condition, whereas non-Hispanic Blacks had the lowest transition rates. The transition rates from healthy to death and one chronic condition to death show that non-Hispanic Blacks have the highest transition rates among all the races, whereas Hispanics have the lowest transition rates. For the transition from multimorbidity to death, non-Hispanic Blacks have the highest transition rates, while non-Hispanic Whites have the lowest transition rates.

The results in Table 1 suggest that the number of people in the United States aged 50 years and older will increase by 61.11% (100% confidence interval 57.2%−66.2%) from 137.25 million (135.64–139.18) in 2020 to 221.13 million (213.24–231.34) in 2050. Remarkably, the number of people aged 80 years and older will increase by 137.26% (116.0%−164.6%), from 16.935 million (16.148–17.863) in 2020 to 40.181 million (34.881–47.272) in 2050. Of the population 50 years and older, the number with at least one chronic disease is estimated to increase by 99.5% (95.1%−107.9%) from 71.522 million (69.065–73.781) in 2020 to 142.66 million (134.74–153.39) by 2050. At the same time, those with multimorbidity are projected to increase 91.16% (79.09%−103.24%) from 7.8304 million (6.5965–9.4853) in 2020 to 14.968 million (11.813–19.277) in 2050. The analysis suggests that by 2035, 35.66% (33.36–36.04) of the adult population 50 years and older will have at least one chronic condition, which is expected to increase to 47.81% (46.09–49.71) by 2050. At the same time, 3.659% (2.905–4.696) of the adult population is expected to have multimorbidity, increasing to 5.017% (3.948–6.481) by 2050. Most individuals with at least one chronic condition (62.75% in 2020 and 58.54% in 2050) or multimorbidity (62.9% in 2020 and 58.9% in 2050) are between the ages of 60 to 79 years. However, individuals aged 80 years and older with one chronic condition and multimorbidity are projected to have the highest increase (244.3% for one chronic condition and 202.7% for multimorbidity) from 2020 to 2050.
The projected number of non-Hispanic Blacks with at least one chronic condition is 8.1994 million (7.6355–8.6193) in 2020 and is expected to increase to 15.2213 million (13.33–16.98) by 2050 [that is a relative change between 2020 and 2050 of 85.64% (74.64–97.01)]. Most non-Hispanic Blacks with one chronic condition and multimorbidity are females between 60 to 79 years old. Similarly, to all the races, the age group with the highest increase is individuals 80 years and older for both one chronic condition and multimorbidity. The number of non-Hispanic Blacks with multimorbidity is projected to increase from 0.9625 million (0.7294–1.2165) in 2020 to 1.7505 million (1.1798–2.489) by 2050, representing a relative increase of 82.87% (61.75%−104.6%).

Hispanic adults 50 years and older with at least one chronic condition are estimated to increase from 11.7996 million (11.125–12.546) in 2020 to 24.732 million (22.214–28.613) by 2050. This change represents an increase of 109.61% (99.67–128.1). Like all races, most Hispanics with one chronic condition are females within the age group of 60 to 79 years, and the age group with the highest increase in the number of people with at least one chronic condition and multimorbidity is individuals aged 80 years and older. Also, the number of Hispanics with multimorbidity is projected to increase from 1.4632 million (1.0713–1.902) in 2020 to 2.9136 million (1.8776–4.2586) by 2050. Most Hispanics with multimorbidity are males between 60 to 79 years old.

Lastly, the number of other races who are not non-Hispanic Whites, Blacks, or Hispanics in the United States with at least one chronic condition is projected to increase from 4.9072 million (4.3938–5.3519) in 2020 to 9.684 million (7.8591–11.6347) by 2050, representing a relative increase of 97.34% (78.87–117.4) from 2020 to 2050. Most of the other races with one chronic condition are females between the ages of 60–79 years. Among the other race, the age group with the highest increase in the number of people with at least one chronic condition and multimorbidity is individuals aged 80 years and older. The projected number of adults 50 years and older categorized as other races with multimorbidity is estimated to increase from 0.5119 million (0.2233–0.8173) in 2020 to 0.8927 million (0.2893–1.9205) by 2050. Most of the other races with multimorbidity are females.
Discussion

The results show that the number of people in the United States aged 50 years and older is projected to increase significantly. Consequently, by 2050, most individuals 50 years and older will have one or more chronic conditions. Most of the population 50 years and older with one or more chronic conditions are projected to be between the ages of 60 to 79 years, and the number of individuals 80 years and older with one or more chronic conditions is expected to more than double from 2020 to 2050. Most individuals 50 years and older with one chronic condition are females, while that with multimorbidity are males.

The insight that the majority of the adult population 50 years and older, across all races, will have at least one chronic condition has health and economic implications. Within the health domain, these insights emphasize the importance of prioritizing the promotion of access to high-quality primary care services that can provide whole-person care that ensures prevention and management of chronic disease care and address evidence-based social determinants of health that increase the risk of developing chronic diseases. Moreover, individual, family and community-oriented health education that highlights the importance of a healthy lifestyle and addresses structural issues that perpetuate health disparities should be a vital part of the health system to change the trajectory of chronic disease. The health education provided to individuals, families and the community and care models offered to the population should emphasize the continuous care models for addressing chronic conditions that help individuals to lead better lives. This health education and care models should focus on self-care (i.e., tasks performed by healthy people to stay healthy) and self-management (i.e., day-to-day tasks undertaken to reduce the impact of chronic disease on physical health status) approaches. These approaches should focus on encouraging the individual to stay healthy and for those with chronic conditions, the ability to manage the symptoms, treatment, physical and psychosocial consequences, and lifestyle changes inherent in living with a chronic condition.

Chronic disease and especially multimorbidity, is associated with increased mortality (6), reduced quality of life, and functional status (2, 7, 8), increased health services use (3, 9), and higher cost of care. As a result, health care systems and policymakers should prioritize cost-effective interventions that have the potential to reduce the cost of chronic disease management to the health care system. Chronic disease is associated with substantial work productivity losses. Thus, policymakers and employers should focus on programs and resource allocation to reduce the incidence and prevalence of chronic disease and absenteeism resulting from chronic diseases to maintain and increase productivity.

The main strength of this paper is the use of 20 years' worth of data to estimate the incidence and prevalence of chronic diseases among the adult population in the United States. The main limitation of this research is, first, the list of chronic diseases included in the Health and Retirement Study is not a comprehensive list of chronic diseases, and the chronic diseases reported in the survey are self-reported. A broader definition of chronic diseases would include more conditions that are not captured in this study. These can potentially underestimate the incidence and prevalence of chronic diseases projected in this study. Hence, the numbers provided in the research should be interpreted within the context of the chronic diseases captured in the survey used herein. Another important limitation is that individuals transitioning to the adult population are assumed to have similar chronic disease transition patterns observed in the Health and Retirement Survey. Lastly, a limitation of the statistic model is that since the data used for this study (Health and Retirement Study) is not an annual survey, we fill in gaps with pseudo-data representing successive years to obtain annual transition probabilities.
Data availability statement

Publicly available datasets were analyzed in this study. This data can be found at: The Health and Retirement Study.
Author contributions

JA conceived and designed the study, developed the multi-state population model to simulate the chronic disease burden among the adult population in the USA, and conducted the analysis and manuscript writing. C-TC conducted the statistical analysis for the transition probabilities using the Health and Retirement Study and developed the R algorithm used for data analytics. All authors contributed to the article and approved the submitted version.
Funding Statement

This research was supported by the Center for Community Health Integration at Case Western Reserve University.
Conflict of interest

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.
Publisher's note

All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article, or claim that may be made by its manufacturer, is not guaranteed or endorsed by the publisher.""",
    """Complementary and alternative healthcare and medical practices (CAM) is a group of diverse medical and health care systems, practices, and products that are not presently considered to be part of conventional medicine. The list of practices that are considered as CAM changes continually as CAM practices and therapies that are proven safe and effective become accepted as the “mainstream” healthcare practices. Today, CAM practices may be grouped within five major domains: alternative medical systems, mind-body interventions, biologically-based treatments, manipulative and body-based methods and energy therapies.

TCM is a system of healing that dates back to 200 B.C. in written form. China, Korea, Japan, India and Vietnam have all developed their own unique versions of traditional medicine. Alternative medicine is commonly categorized together with complementary medicine under the umbrella term “complementary and alternative medicine”. Complementary medicine refers to therapies that complement traditional western (or allopathic) medicine and is used together with conventional medicine, and alternative medicine is used in place of conventional medicine. Alternative medicine refers to therapeutic approaches taken in place of traditional medicine and used to treat or ameliorate disease. Integrative medicine refers to combining complementary treatments with conventional care. The basic philosophy of complementary and alternative medicine include holistic care, which focuses on treating a human being as a whole person.

Examples of complementary and alternative medicine healing systems include Ayurveda, which originated in India more than 5,000 years ago, emphasizes a unique cure per individual circumstances. It incorporates treatments including yoga, meditation, massage, diet and herbs; Homeopathy uses minute doses of a substance that causes symptoms to stimulate the body’s self-healing response. Naturopathy focuses on non-invasive treatments to help your body do its own healing. Ancient medicines (complementary and alternative medicine treatments) include Chinese, Asian, Pacific Islander, American Indian and Tibetan practices.

Conventional medicine relies on methods proved to be safe and effective with carefully designed trials and research. But, many complementary and alternative treatments lack solid research on which to base sound decisions. The dangers and possible benefits of many complementary and alternative treatments remain unproved.

While the whole medical systems differ in their philosophical approaches to the prevention and treatment of disease, they share a number of common elements. These systems are based on the belief that one’s body has the power to heal itself. Healing often involves marshalling multiple techniques that involve the mind, body and spirit. Treatment is often individualized and dependent on the presenting symptoms.

Basic principles of integrative medicine include a partnership between the patient and the practitioner in the healing process, the appropriate use of conventional and alternative methods to facilitate the body’s innate healing response, the consideration of all factors that influence health, wellness and disease, including mind, spirit and community as well as body, a philosophy that neither rejects conventional medicine nor accepts alternative medicine uncritically, recognition that good medicine should be based in good science, inquiry driven and open to new paradigms, the use of natural, less invasive interventions whenever possible, the broader concepts of promotion of health and the prevention of illness as well as the treatment of disease. Studies are underway to determine the safety and usefulness of many CAM practices. As research continues, many of the answers about whether these treatments are safe or effective will become clearer.

The use of alternative medicine appears to be increasing. A 1998 study showed that the use of alternative medicine in the USA had risen from 33.8% in 1990 to 42.1% in 1997 [1]. The most common CAM therapies used in the USA in 2002 were prayer (45.2%), herbalism (18.9%), breathing meditation (11.6%), meditation (7.6%), chiropractic medicine (7.5%), yoga (5.1%), body work (5.0%), diet-based therapy (3.5%), progressive relaxation (3.0%), mega-vitamin therapy (2.8%) and visualization (2.1%) [2, 3]. In the United Kingdom, limited data seem to support the idea that CAM use in the United Kingdom is high and is increasing.

Increasing numbers of medical colleges have started offering courses in alternative medicine. Accredited Naturopathic colleges and universities are increasing in number and popularity in the USA. They offer the most complete medical training in complimentary medicines that is available today [4, 5]. In Britain, no conventional medical schools offer courses that teach the clinical practice of alternative medicine. However, alternative medicine is taught in several unconventional schools as part of their curriculum. Teaching is based mostly on theory and understanding of alternative medicine, with emphasis on being able to communicate with alternative medicine specialists.

Naturopathy (naturopathic medicine) is a whole medical system that has its roots in Germany. It was developed further in the late 19th and early 20th centuries in the United States, where today it is part of CAM. Naturopathy aims to support the body’s ability to heal itself through the use of dietary and lifestyle changes together with CAM therapies such as herbs, massage and joint manipulation. Naturopathy is a whole medical system. It views disease as a manifestation of alterations in the processes by which the body naturally heals itself and emphasizes health restoration rather than disease treatment. Naturopathic physicians employ an array of healing practices, including diet and clinical nutrition, homeopathy, acupuncture, herbal medicine, hydrotherapy, spinal and soft-tissue manipulation, physical therapies involving electric currents, ultrasound and light therapy, therapeutic counseling and pharmacology. Today, naturopathy is practiced in a number of countries, including the United States, Canada, Great Britain, Australia and New Zealand.

The acupuncture is being practiced for relief or the prevention of pain and for various other health conditions. Preclinical studies have documented acupuncture’s effects, but they have not been able to fully explain how acupuncture works within the framework of the western system of medicine.

Ayurveda, which literally means “the science of life”, is a natural healing system developed in India. It is a comprehensive system of medicine that places equal emphasis on the body, mind and spirit, and strives to restore the innate harmony of the individual. Some of the primary Ayurvedic treatments include diet, exercise, meditation, herbs, massage, exposure to sunlight, and controlled breathing, Ayurvedic medications have the potential to be toxic. Most Ayurvedic medications consist of combinations of herbs and other medicines, so it can be challenging to know which ones are having an effect and why.

Other traditional medical systems have been developed by Native American, Aboriginal, African, Middle-Eastern, Tibetan, Central and South American cultures.

Homeopathy is a system of medical theory and practice. Its founder, German physician Samuel Christian Hahnemann (1755–1843), hypothesized that one can select therapies on the basis of how closely symptoms produced by a remedy match the symptoms of the patient’s disease. He called this the “principle of similars”. Since homeopathy is administered in minute or potentially non-existent material dosages, there is an a priori skepticism in the scientific community about its efficacy [6–9].

Traditional oriental medicine emphasizes the proper balance or disturbances of qi (pronounced chi), or vital energy, in health and disease, respectively. Traditional oriental medicine consists of a group of techniques and methods, including acupuncture, herbal medicine, oriental massage and qi gong (a form of energy therapy described more fully below).

Naturopathy (naturopathic medicine) is a whole medical system that has its roots in Germany. It was affect bodily function and symptoms. Only a subset of mind-body interventions is considered CAM. Many that have a well-documented theoretical basis, for example, patient education and cognitive-behavioral approaches are now considered “mainstream”. On the other hand, meditation, certain uses of hypnosis, dance, music and art therapy and prayer and mental healing are categorized as complementary and alternative.

Biofeedback is a type of mind-body therapy. Using feedback from a variety of monitoring procedures and equipment, a biofeedback specialist will try to teach you to control certain involuntary body responses, such as: brain activity, blood pressure, muscle tension and heart rate. Biofeedback has been shown to be helpful in treating several medical conditions, including asthma, Raynaud’s disease, irritable bowel syndrome, incontinence, headaches, cardiac arrhythmias, high blood pressure, epilepsy, etc.

The term meditation refers to a variety of techniques or practices intended to focus or control attention. Most of them are rooted in Eastern religious or spiritual traditions. These techniques have been used by many different cultures throughout the world for thousands of years.

People have used prayer and other spiritual practices for their own and others’ health concerns for thousands of years. Scientific investigation of these practices has begun quite recently, however, to better understand whether they work; if so, how; and for what diseases/conditions and populations. Many Americans are using prayer and other spiritual practices. Prayer is the therapy most commonly used among all the CAM therapies.

Manipulative and body-based practices include methods that are based on manipulation and/or the movement of the body. For example, chiropractors focus on the relationship between structure (primarily the spine) and function, and how that relationship affects the preservation and restoration of health, using manipulative therapy as an integral treatment tool. Massage therapists manipulate the soft tissues of the body to normalize those tissues.

Energy therapies focus either on energy fields originating within the body (biofields) or those from other sources (electromagnetic fields). Biofield therapies are intended to affect the energy fields, whose existence is not yet experimentally proven, that surround and penetrate the human body. Some forms of energy therapy manipulate biofields by applying pressure and/or manipulating the body by placing the hands in, or through, these fields. Examples include Qi gong, Reiki, Prana and Therapeutic Touch. Bioelectromagnetic-based therapies involve the unconventional use of electromagnetic fields, such as pulsed fields, magnetic fields or alternating current or direct current fields, to, for example, treat asthma or cancer, or manage pain and migraine headaches.

Hypnosis is an altered state of consciousness. Hypnotherapy has the potential to help relieve the symptoms of a wide variety of diseases and conditions. It can be used independently or along with other treatments.

Natural and biologically-based practices, interventions and products refer to the use of dietary supplements and include herbal, special dietary, orthomolecular and individual biological therapies. Examples include botanicals, animal-derived extracts, vitamins, minerals, fatty acids, amino acids, proteins and prebiotics, Thousands of studies of various dietary supplements have been performed. However, no single supplement has been proven effective in a compelling way.

In India, which is the home of several alternative systems of medicines, Ayurveda, Siddha, Unani and Homeopathy are licenced by the government, despite the lack of reputable scientific evidence. Naturopathy will also be licensed soon because several universities now offer bachelors degrees in it. Other activities such as Panchakarma and massage therapy related to Ayurveda are also licensed by the government now [10].

About half the general population in developed countries uses CAM [10]. A survey released in May 2004 by the National Center for Complementary and Alternative Medicine, part of the National Institutes of Health in the United States, found that in 2002, 36% of Americans used some form of alternative therapy in the past 12 months, 50% in a lifetime—a category that included yoga, meditation, herbal treatments and the Atkins diet. The majority of individuals (54.9%) used CAM in conjunction with conventional medicine. Most people use CAM to treat and/or prevent musculoskeletal conditions or other conditions associated with chronic or recurring pain. Women were more likely than men to use CAM. The largest sex differential is seen in the use of mind-body therapies including prayer specifically for health reasons [2, 3]. If prayer was counted as an alternative therapy, the figure rose to 62.1%. 25% of people who use CAM do so because a medical professional suggested it [11]. A British telephone survey by the BBC of 1,209 adults in 1998 shows that around 20% of adults in Britain had used alternative medicine in the past 12 months.

Advocates of alternative medicine hold that the various alternative treatment methods are effective in treating a wide range of major and minor medical conditions, and contend that recently published research (Michalsen, 2003; Gonsalkorale, 2003; Berga, 2003) proves the effectiveness of specific alternative treatments [6–9].

Evidence-based medicine (EBM) applies the scientific method to medical practice, and aims for the ideal that healthcare professionals should make “conscientious, explicit, and judicious use of current best evidence” in their everyday practice. Although advocates of alternative medicine acknowledge that the placebo effect may play a role in the benefits that some receive from alternative therapies, they point out that this does not diminish their validity. They believe that alternative medicine may provide health benefits through patient empowerment, by offering more choices to the public. Researchers who judge treatments using the scientific method are concerned by this viewpoint, since it fails to address the possible inefficacy of alternative treatments.

As long as alternative treatments are used alongside conventional treatments, the majority of medical doctors find most forms of complementary medicine acceptable. Consistent with previous studies, the CDC recently reported that the majority of individuals in the United States (i.e., 54.9%) used CAM in conjunction with conventional medicine.

The issue of alternative medicine interfering with conventional medical practices is minimized when it is turned to only after conventional treatments have been exhausted. Many patients feel that alternative medicine may help in coping with chronic illnesses for which conventional medicine offers no cure, only management. Classifying treatments need to be based on the objectively verifiable criteria of the scientific method evidence-based medicine, i.e. scientifically proven evidence of efficacy (or lack thereof), and not on the changing curricula of various medical schools or social sphere of usage [12].

Since many alternative remedies have recently found their way into the medical mainstream, there cannot be two kinds of medicine - conventional and alternative. There is only medicine that has been adequately tested and medicine that has not, medicine that works and medicine that may or may not work. Once a treatment has been tested rigorously, it no longer matters whether it was considered alternative at the outset. If it is found to be reasonably safe and effective, it will be accepted [13].

It is argued that there is no alternative medicine. There is only scientifically proven, evidence-based medicine supported by solid data or unproven medicine, for which scientific evidence is lacking. Whether a therapeutic practice is “Eastern” or “Western”, is unconventional or mainstream, or involves mind-body techniques or molecular genetics is largely irrelevant except for historical purposes and cultural interest. As believers in science and evidence, we must focus on fundamental issues—namely, the patient, the target disease or condition, the proposed or practiced treatment, and the need for convincing data on safety and therapeutic efficacy [14]. The Cochrane Collaboration [15] and Edzard Ernst [16] agree that all treatments, whether “mainstream” or “alternative”, ought to be held to standards of the scientific method.

Many forms of alternative medicine are rejected by conventional medicine because the efficacy of the treatments has not been demonstrated through double-blind randomized controlled trials; in contrast, conventional drugs reach the market only after such trials have proved their efficacy. A person may attribute symptomatic relief to an otherwise ineffective therapy due to the placebo effect, the natural recovery from or the cyclical nature of an illness (the regression fallacy), or the possibility that the person never originally had a true illness [17]. CAM proponents point out this may also apply in cases where conventional treatments have been used. To this, CAM critics point out that this does not account for conventional medical success in double blind clinical trials.

People should be free to choose whatever method of healthcare they want, but stipulate that people must be informed as to the safety and efficacy of whatever method they choose. People who choose alternative medicine may think they are choosing a safe, effective medicine, while they may only be getting quack remedies. Grapefruit seed extract is an example of quackery when multiple studies demonstrate its universal antimicrobial effect is due to synthetic antimicrobial contamination [18, 19].

Those who have had success with one alternative therapy for a minor ailment may be convinced of its efficacy and persuaded to extrapolate that success to some other alternative therapy for a more serious, possibly life-threatening illness. For this reason, critics contend that therapies that rely on the placebo effect to define success are very dangerous. Scientifically unsupported health practices can lead individuals to forgo effective treatments and this can be referred to as “opportunity cost”. Individuals who spend large amounts of time and money on ineffective treatments may be left with precious little of either, and may forfeit the opportunity to obtain treatments that could be more helpful. More research must be undertaken to prove the effectiveness of complimentary therapies before they can be incorporated in formal medical practice. Sufficient evidence is required for biological or clinical plausibility in order to justify the investment of time and energy in exploring the merits of alternative medicine. After all, human life is precious and no chances can be taken to comprise the health of any individual.""",
]
labels = [0.9, 0.9, 0.9]  # Continuous confidence scores

# Split into training and evaluation datasets
train_texts = texts[:3]
train_labels = labels[:3]
eval_texts = texts[3:]
eval_labels = labels[3:]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = ScientificDataset(train_texts, train_labels, tokenizer)
eval_dataset = ScientificDataset(eval_texts, eval_labels, tokenizer)

# Load pre-trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1
)  # Set num_labels to 1 for regression
model.to(device)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    save_total_limit=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Saving the model and tokenizer
model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")


# Inference function
def predict_confidence_with_source(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(**inputs)
    scores = torch.sigmoid(
        outputs.logits
    )  # Use sigmoid for regression output in confidence
    confidence = scores.item()

    # Placeholder for source tracking logic
    sources = "Source details would be implemented here"

    return confidence, sources


# FastAPI setup
app = FastAPI()

class TextInput(BaseModel):
    text: str
    
@app.post("/predict_confidence")
def get_confidence(input_data: TextInput):
    statement = input_data.text
    confidence, sources = predict_confidence_with_source(statement)
    return {"confidence": confidence, "sources": sources}


# For running the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Function to prepare paper sections (for demonstration purposes)
def prepare_paper_for_training(paper_text):
    # Example preprocessing, implement according to your needs
    sections = paper_text.split("\n")  # Example of simple splitting
    return sections  # This would be further processed and fed into datasets
