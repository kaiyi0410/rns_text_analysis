library(jsonlite)
library(tidyverse)
library(quanteda)
library(quanteda.textstats)

setwd('/Users/ziyilow/Documents/MSc/Tackling_Sust_Challenges/4_RNS')

df = read_csv('/Users/ziyilow/Documents/MSc/Tackling_Sust_Challenges/4_RNS/final_df_sampled.csv') %>%
  select(-"...1")

#### ====================== Tokenising ====================== ####

# tokenise, remove punctuation, symbols, numbers etc
token = tokens(df$text_body, what = "word",
             remove_punct = TRUE,
             remove_symbols = TRUE,
             remove_numbers = TRUE,
             remove_url = TRUE,
             remove_hyphens = FALSE,
             verbose = TRUE, 
             include_docvars = TRUE)

# convert all to lowercase
token <- tokens_tolower(token)

# remove stop words
token <- tokens_select(tok, stopwords("english"), 
                       selection = "remove", padding = FALSE)

# number of tokens (number of words)
ntoken(token)
# number of types (number of unique words)
ntype(token)


#### ================= Readability metrics ================= #####

# mean sentence length, mean word syllables, Gunning-Fog index 
readability <- textstat_readability(df$text_body, c("meanSentenceLength",
                                                    "meanWordSyllables", "FOG"), 
                                    remove_symbols = TRUE,
                                    remove_numbers = TRUE,
                                    remove_url = TRUE,
                                    min_sentence_length = 1, 
                                    max_sentence_length = 10000,
                                    intermediate = FALSE)[,2:4]

# check NAs
sum(is.na(readability$meanSentenceLength))
which(is.na(readability$meanSentenceLength))


## ====================== Similarity Network ===================== ###

# company metadata 
comp_mtd <- fromJSON('/Volumes/UoE_EnMa023/MTHM604_Data/04_KernowAM/CompanyData/company_metadata.json')

# filling in NA's for each list element 
comp_mtd <- lapply(comp_mtd, function(x){	
  Map(function(z){
    ifelse(is.null(z), NA, z)},  
    x) })
# converting to data frame 
cmtd <- data.frame(lapply(comp_mtd, function(x) Reduce(c, x)))

# frequency of companies appearing in subset
cs <- as.data.frame(table(df$RIC))

# sort by most frequent appearance and choose 50 most frequent 
cs2 <- cs %>% 
  arrange(desc(Freq)) %>%
  slice(1:50)

# add industry of each company
n_distinct(cmtd$TRBC.Economic.Sector.Name)
cmtd2 <- cmtd %>% filter(RIC %in% cs2$Var1 == TRUE)
cs2 <- rename(cs2, "RIC" = "Var1")
ind <- merge(x = cs2, y = cmtd2[ , c("RIC", "TRBC.Economic.Sector.Name")], 
             by = "RIC")

# add sector name to original subset
df <- merge(x = df, y = cmtd[ , c("RIC", "TRBC.Economic.Sector.Name")], 
            by = "RIC", all.x = TRUE)

# filter rns for those 50 companies
rns50 <- df %>% 
  filter(RIC %in% ind$RIC == TRUE) %>%
  select(c("date", "RIC", "text_body", "multi_horizon_return"))

# join RNS text by company
joined <- aggregate(x=list(multiRNS=rns50$text_body), 
                    by=list(group=rns50$RIC), paste, collapse="")

# new df with IDs to make corpus
rns50corp <- rns50 %>% 
  select(c("RIC", "text_body")) %>%
  mutate(id = c(1:2147))

corp <- corpus(rns50corp, 
               text_field = "text_body", docid_field = "id")

# tokenise and remove stopwords
dfmat_rns <- corp %>% 
  tokens(remove_punct = TRUE, remove_url = TRUE, 
         remove_symbols = TRUE, remove_numbers = TRUE)
dfmat_rns <- dfm(dfmat_rns) %>%
  dfm_remove(pattern = stopwords("en"))

# check number of documents
ndoc(dfmat_rns)
# most frequent features
topfeatures(dfmat_rns, n = 10, groups = RIC)

# combine documents by company
dfmat_comps <- dfm_group(dfmat_rns, groups = RIC)

# calculate similarity
mat <- as.matrix(textstat_simil(dfmat_comps))


## ====================== Plotting ======================== ###

library(igraph)
library(RColorBrewer)

# remove lower similarity pairs for less cluttered plot
mat2 <- mat
mat2[mat2<0.6] <- 0 # ranges from 0 to 0.87

# igraph object
network <- graph_from_adjacency_matrix(mat2, weighted=T, 
                                       mode="undirected", diag=F)

# check basic chart
plot(network, layout= layout.fruchterman.reingold, main="fruchterman.reingold")

# colour brewer
coul <- brewer.pal(nlevels(as.factor(ind$TRBC.Economic.Sector.Name)), "RdYlGn")

# map colour to industry
my_color <- coul[as.numeric(as.factor(ind$TRBC.Economic.Sector.Name))]

# plot
par(bg = "white", mar=c(1,1,1,1))
set.seed(4)
plot(network, 
     vertex.size=12,
     vertex.color=my_color, 
     vertex.label.cex=0.7,
     vertex.label.color="gray17",
     vertex.label.family = "sans",
     vertex.frame.color="transparent",
     edge.color = "gray30",
     edge.width = 1.5
)

# legend
legend(x=-1.75, y=1, 
       legend=levels(as.factor(ind$TRBC.Economic.Sector.Name)),
       col = coul,
       bty = "n", pch=20 , pt.cex = 4, cex = 1,
       text.col="black" , horiz = F)


## ====================== Binomial GLM ======================== ###

library(ggfortify)
library(lme4)
library(DescTools)

# full df with metrics
df3 <- read_csv('rns_final_metrics.csv') %>%
  select(-"...1")

# subset by positive and negative returns 
pos_sub <- df3 %>% filter(ret20 > 0)
neg_sub <- df3 %>% filter(ret20 < 0)


## Positive subset
# full model
m <- glm(ret20 ~ Subjectivity, family = quasibinomial, data = pos_sub)
summary(m)
# null model
mn <- glm(ret20 ~ 1, family = quasibinomial, data = pos_sub)
# likelihood ratio test
anova(mn, m, test = 'Chisq')
# check residuals 
autoplot(m)
# r2
with(summary(m), 1 - deviance/null.deviance)

# scatter plot
ggplot(pos_ss, aes(Subjectivity, ret20)) +
  geom_point(size = 7, color = '#F7CE5D') + stat_smooth(method = 'lm', size = 5, 
                                                        colour = 'gray17') +
  xlab("Subjectivity") + ylab("20-day Returns") +
  theme(panel.grid = element_blank(),
        panel.background = element_rect(fill = 'white'),
        panel.border = element_rect(fill = 'NA', colour = 'white'),
        axis.text = element_text(colour = 'black', size = 30),
        axis.title = element_text(colour = 'black', face = 'bold', size = 36))


## Negative subset
# flip signs
neg_sub$neg_ret <- -neg_sub$ret20
neg_sub2 <- filter(neg_sub, GFI <21)

# full model
m <- glm(neg_ret ~ Subjectivity + GFI, family = quasibinomial, data = neg_sub2)
summary(m)
# subjectivity null model
mn <- glm(neg_ret ~ GFI, family = quasibinomial, data = neg_sub2)
# likelihood ratio test
anova(mn, m, test = 'Chisq')

# GFI null model
mn2 <- glm(neg_ret ~ Subjectivity, family = quasibinomial, data = neg_sub)
# LRT
anova(m, mn2, test = 'Chisq')
# check resids
autoplot(m)
# r2
with(summary(m), 1 - deviance/null.deviance)

# plotting
ggplot(neg_sub, aes(Subjectivity, ret20)) +
  geom_point(size = 7, color = '#ED5F65') + stat_smooth(method = 'lm', size = 5, colour = 'gray17') +
  xlab("Gunning-Fog Index") + ylab("20-day Returns") +
  theme(panel.grid = element_blank(),
        panel.background = element_rect(fill = 'white'),
        panel.border = element_rect(fill = 'NA', colour = 'white'),
        axis.text = element_text(colour = 'black', size = 30),
        axis.title = element_text(colour = 'black', face = 'bold', size = 36))






