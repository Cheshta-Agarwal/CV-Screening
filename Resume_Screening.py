import pandas as pd
import re

# Load JSON data
df = pd.read_json('fixed_resume_data.json')

# --- Data Extraction Functions ---
def extract_info(annotations, label):
    """Safely extract text from annotations for a given label"""
    items = []
    if isinstance(annotations, list):
        for annotation in annotations:
            if annotation.get('label') == [label]:
                for point in annotation.get('points', []):
                    text = point.get('text', '').strip()
                    if text:  # Skip empty strings
                        items.append(text)
    return items

# --- Core Field Extraction ---
df['name'] = df['annotation'].apply(lambda x: extract_info(x, 'Name'))
df['skills'] = df['annotation'].apply(lambda x: extract_info(x, 'Skills'))
df['degrees'] = df['annotation'].apply(lambda x: extract_info(x, 'Degree'))
df['colleges'] = df['annotation'].apply(lambda x: extract_info(x, 'College Name'))
df['locations'] = df['annotation'].apply(lambda x: extract_info(x, 'Location'))
df['companies'] = df['annotation'].apply(lambda x: extract_info(x, 'Companies worked at'))
df['designations'] = df['annotation'].apply(lambda x: extract_info(x, 'Designation'))

# --- Experience Extraction ---
def extract_experience(text):
    """Extract years of experience using regex"""
    matches = re.findall(
        r'(\d+\.?\d*)\s*(?:years?|yrs?)\s*(?:of\s*experience)?',
        str(text),
        flags=re.IGNORECASE
    )
    return float(matches[0]) if matches else None

df['experience_years'] = df['content'].apply(extract_experience)

# --- Text Normalization ---
def normalize_text_list(text_list):
    """Lowercase, strip whitespace, and deduplicate list items"""
    if isinstance(text_list, list):
        return list(set([str(item).lower().strip() for item in text_list if item]))
    return []

# Apply to all text-based columns
text_columns = ['skills', 'degrees', 'colleges', 'companies', 'designations']
for col in text_columns:
    df[col] = df[col].apply(normalize_text_list)

# --- Handle List Columns ---
# Explode list columns for analysis-ready format
exploded_df = df.explode('skills').explode('designations').dropna(subset=['skills', 'designations'])

# --- Save Processed Data ---
exploded_df.to_csv('processed_resumes.csv', index=False)
print("Pre-processing complete. Saved to processed_resumes.csv")

import pandas as pd
import re
import numpy as np

# Load the CSV file
df = pd.read_csv('processed_resumes.csv')

# --- 1. Clean List-like Columns ---
def clean_list_string(list_str):
    """Safely convert string representations to lists"""
    if pd.isna(list_str) or list_str == '[]':
        return []
    if isinstance(list_str, list):
        return list_str
    # Handle both quoted and unquoted items
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", str(list_str).strip('[]'))
    # Flatten matches and filter empty
    cleaned_items = []
    for item in items:
        combined = item[0] or item[1] or item[2]
        if combined:
            cleaned_items.append(combined.strip())
    return cleaned_items

list_columns = ['skills', 'degrees', 'colleges', 'companies', 'designations']
for col in list_columns:
    df[col] = df[col].apply(clean_list_string)

# --- 2. Handle Missing Values ---
df['experience_years'] = pd.to_numeric(df['experience_years'], errors='coerce').fillna(0)
df['locations'] = df['locations'].fillna('Unknown')

# --- 3. Text Normalization ---
def normalize_text(text):
    """Handle both strings and list elements"""
    if isinstance(text, list):
        return [normalize_text(item) for item in text]
    if pd.isna(text):
        return text
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)      # Collapse spaces
    return text

# Apply normalization
for col in df.columns:
    if col in list_columns:
        df[col] = df[col].apply(lambda x: [normalize_text(item) for item in x] if isinstance(x, list) else [])
    elif df[col].dtype == object:
        df[col] = df[col].apply(normalize_text)

# --- 4. Deduplicate List Items ---
for col in list_columns:
    df[col] = df[col].apply(lambda x: list(set(x)) if x else [])

# --- 5. Standardize Common Terms ---
role_mapping = {
    r'senior\s+software': 'software engineer',
    r'devops': 'devops engineer',
    r'data\s+scientist': 'data scientist',
    r'analyst': 'data analyst'
}

skill_mapping = {
    r'java\s*\w*': 'java',
    r'python\s*\w*': 'python',
    r'sql\s*\w*': 'sql',
    r'aws\s*\w*': 'aws',
    r'machine\s+learning': 'machine learning'
}

def standardize_terms(text_list, mapping):
    """Apply regex-based standardization"""
    if not isinstance(text_list, list):
        return text_list
    standardized = []
    for item in text_list:
        for pattern, replacement in mapping.items():
            if re.search(pattern, item, flags=re.IGNORECASE):
                standardized.append(replacement)
                break
        else:
            standardized.append(item)
    return standardized

for col in ['designations']:
    df[col] = df[col].apply(lambda x: standardize_terms(x, role_mapping))

for col in ['skills']:
    df[col] = df[col].apply(lambda x: standardize_terms(x, skill_mapping))

# --- 6. Filter Valid Records ---
df = df[df['skills'].apply(len) > 0]
df = df[df['designations'].apply(len) > 0]

# --- 7. Create Exploded Views ---
exploded_skills = df.explode('skills')
exploded_designations = df.explode('designations')

# --- 8. Save Results ---
df.to_csv('cleaned_resumes.csv', index=False)
exploded_skills.to_csv('exploded_skills.csv', index=False)
exploded_designations.to_csv('exploded_designations.csv', index=False)

print("Data cleaning complete. Output files:")
print("- cleaned_resumes.csv")
print("- exploded_skills.csv")
print("- exploded_designations.csv")

# Flatten skills and count occurrences
all_skills = [skill for sublist in df['skills'] for skill in sublist]
skill_counts = pd.Series(all_skills).value_counts().head(20)

# Plot top skills
plt.figure(figsize=(12, 8))
sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='viridis')
plt.title('Top 20 Most Common Skills')
plt.xlabel('Count')
plt.ylabel('Skill')
plt.tight_layout()
plt.show()

# Degree distribution
degree_counts = pd.Series([deg for sublist in df['degrees'] for deg in sublist]).value_counts().head(10)

plt.figure(figsize=(10, 6))
degree_counts.plot(kind='barh', color='skyblue')
plt.title('Top 10 Degrees')
plt.xlabel('Count')
plt.ylabel('Degree')
plt.show()

# Top colleges
college_counts = pd.Series([col for sublist in df['colleges'] for col in sublist]).value_counts().head(10)

plt.figure(figsize=(10, 6))
college_counts.plot(kind='bar', color='salmon')
plt.title('Top 10 Colleges/Universities')
plt.xlabel('College')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['experience_years'].dropna(),
             bins=15,
             kde=True,
             color='purple',
             edgecolor='black')
plt.title('Distribution of Years of Experience', fontsize=14, pad=20)
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

top_skills = ['java', 'python', 'sql', 'aws', 'machine learning']
for skill in top_skills:
    df[f'has_{skill}'] = df['skills'].apply(
        lambda x: 1 if skill in str(x).lower() else 0
    )
plt.figure(figsize=(10, 6))
sns.boxplot(x='has_java', y='experience_years',data=df)
plt.title('Years of Experience for Resumes With/Without Java', fontsize=14, pad=15)
plt.xlabel('Java Skill Presence', fontsize=12)
plt.ylabel('Years of Experience', fontsize=12)
plt.xticks([0, 1], ['Without Java', 'With Java'])
plt.grid(axis='y', alpha=0.3)
plt.show()

# Location distribution
location_counts = pd.Series([loc for sublist in df['locations'] for loc in sublist]).value_counts().head(10)

plt.figure(figsize=(10, 6))
location_counts.plot(kind='bar', color='green')
plt.title('Top 10 Locations')
plt.xlabel('Location')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# --- 2. Enhanced Heatmap Visualization ---
plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    matrix,
    annot=True,
    fmt='.0f',  # Integer percentages
    cmap='YlGnBu',
    linewidths=0.5,
    linecolor='lightgray',
    cbar_kws={'label': 'Skill Prevalence (%)', 'shrink': 0.8},
    annot_kws={'size': 9, 'color': 'black'}
)

# Customize appearance
plt.title('Role-Skill Matching Analysis\n(% of Professionals in Each Role with Skill)',
          pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Technical Skills', fontsize=12, labelpad=10)
plt.ylabel('Job Roles', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add value interpretation guide
plt.text(0.5, -0.15,
         "Values show percentage of professionals in each role who list the skill",
         ha='center', va='center',
         transform=ax.transAxes,
         fontsize=10,
         color='gray')

plt.tight_layout()
plt.savefig('role_skill_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 3. Top Skills Per Role (Alternative View) ---
for role in matrix.index[:3]:  # Show top 3 roles
    plt.figure(figsize=(10, 4))
    role_skills = matrix.loc[role].sort_values(ascending=False).head(8)

    sns.barplot(
        x=role_skills.values,
        y=role_skills.index,
        palette='Blues_r',
        edgecolor='black'
    )

    plt.title(f'Top Skills for {role.title()}', fontsize=13, pad=15)
    plt.xlabel('Percentage of Professionals', fontsize=11)
    plt.ylabel('')
    plt.xlim(0, 100)

    # Add value labels
    for i, v in enumerate(role_skills.values):
        plt.text(v + 2, i, f"{v:.0f}%",
                 color='black',
                 va='center',
                 fontsize=10)

    plt.tight_layout()
    plt.show()

# Top companies
company_counts = pd.Series([comp for sublist in df['companies'] for comp in sublist]).value_counts().head(15)

plt.figure(figsize=(10, 8))
company_counts.plot(kind='barh', color='orange')
plt.title('Top 15 Companies Worked At')
plt.xlabel('Count')
plt.ylabel('Company')
plt.show()

# Top designations
designation_counts = pd.Series([des for sublist in df['designations'] for des in sublist]).value_counts().head(15)

plt.figure(figsize=(10, 8))
designation_counts.plot(kind='barh', color='blue')
plt.title('Top 15 Designations')
plt.xlabel('Count')
plt.ylabel('Designation')
plt.show()


# --- 1. Calculate Correlations ---
corr_matrix = skill_df.corr()

# --- 3. Enhanced Correlation Heatmap ---
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 9}
)

# Customize appearance
plt.title('Skill Correlation Matrix\n(With Experience Years)',
          pad=20, fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Add interpretation note
plt.text(0.5, -0.15,
         "Values show Pearson correlation coefficients (-1 to +1)",
         ha='center', va='center',
         transform=plt.gca().transAxes,
         fontsize=10,
         color='gray')

plt.tight_layout()

plt.show()

import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# === 1. Ultra-Robust Data Loading ===
def load_resumes(file_path):
    resumes = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and 'content' in data:
                    resumes.append(data)
            except:
                continue
    return resumes

# === 2. Fail-Safe Text Processing ===
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove all unwanted sections
    text = re.sub(r'(?i)\b(email|phone|location|indeed|linkedin|http[s]?://\S+)\b.*?\n', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text if len(text) > 20 else ""  # Minimum 20 characters

# === 3. Smart Label Extraction ===
def get_label(resume):
    if not isinstance(resume.get('annotation'), list):
        return None

    for ann in resume['annotation']:
        if not isinstance(ann, dict):
            continue
        if 'Designation' in str(ann.get('label', '')):
            points = ann.get('points', [{}])
            if points and isinstance(points[0], dict):
                label = points[0].get('text', '').strip()
                if label:
                    return label
    return None

def categorize_label(label):
    if not label:
        return "Other"

    label = label.lower()
    role_map = [
        (['engineer', 'developer', 'programmer'], 'Engineer'),
        (['analyst', 'scientist', 'bi'], 'Analyst'),
        (['consultant', 'advisor', 'specialist'], 'Consultant'),
        (['manager', 'lead', 'head'], 'Manager'),
        (['teacher', 'professor', 'lecturer'], 'Educator')
    ]

    for keywords, role in role_map:
        if any(kw in label for kw in keywords):
            return role
    return "Other"

# === 4. Main Processing ===
resumes = load_resumes('fixed_resume_data.jsonl')

data = []
for resume in resumes:
    try:
        content = resume.get('content', '')
        if not content:
            continue

        # Get clean text from entire resume (more reliable than section splitting)
        text = clean_text(content)
        if not text:
            continue

        # Get and validate label
        raw_label = get_label(resume)
        label = categorize_label(raw_label)
        if not label:
            continue

        data.append({'text': text, 'label': label})
    except:
        continue

if not data:
    raise ValueError("No valid resumes found after processing")

df = pd.DataFrame(data)

# === 5. Vectorization with Fallbacks ===
try:
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=3
    )
    X = vectorizer.fit_transform(df['text'])
except:
    try:
        vectorizer = CountVectorizer(
            min_df=2,
            max_features=3000
        )
        X = vectorizer.fit_transform(df['text'])
    except:
        raise ValueError("Failed to vectorize text data")

# === 6. Label Encoding ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# === 7. Auto-Balancing ===
if len(set(y)) > 1:  # Only balance if we have multiple classes
    class_counts = pd.Series(y).value_counts()
    if max(class_counts) / min(class_counts) > 3:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

# === 8. Model Training ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Try multiple models with safe defaults
models = {
    'SVM': LinearSVC(class_weight='balanced', max_iter=10000),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

best_model = None
best_score = 0

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} accuracy: {score:.2f}")
        if score > best_score:
            best_score = score
            best_model = model
    except:
        continue

if best_model is None:
    raise ValueError("All models failed to train")

# === 9. Results ===
print(f"\nBest Model: {type(best_model).__name__}")
print(f"Accuracy: {best_score:.2f}")

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

# Download NLTK resources (only needed first time)
nltk.download('stopwords')
nltk.download('wordnet')

# Suppress warnings
warnings.filterwarnings('ignore')

class SkillMatcher:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.skill_keywords = {
            'programming': ['java', 'python', 'c++', 'c#', 'javascript', 'php', 'ruby', 'swift', 'kotlin', 'scala'],
            'database': ['sql', 'mysql', 'oracle', 'mongodb', 'postgresql', 'nosql', 'cassandra', 'redis'],
            'web': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'django', 'flask', 'node.js'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'aws', 'azure', 'gcp'],
            'data_science': ['python', 'r', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn'],
            'cloud': ['aws', 'azure', 'google cloud', 'gcp', 'cloud computing'],
            'networking': ['tcp/ip', 'dns', 'vpn', 'firewall', 'load balancing'],
            'testing': ['selenium', 'junit', 'testng', 'pytest', 'qa', 'quality assurance'],
            'security': ['cybersecurity', 'encryption', 'penetration testing', 'ethical hacking'],
            'mobile': ['android', 'ios', 'react native', 'flutter']
        }
        self.model = None
        self.vectorizer = None
        self.job_descriptions = []

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize and remove stopwords
        tokens = [word for word in text.split() if word not in self.stop_words]

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    def extract_skills(self, resume_data):
        """Extract skills from resume data"""
        skills = []

        for resume in resume_data:
            content = resume.get('content', '')
            annotations = resume.get('annotation', [])

            # Extract from skills section
            skill_sections = [a for a in annotations if a['label'] == ['Skills']]
            skill_texts = [content[a['points'][0]['start']:a['points'][0]['end']] for a in skill_sections]

            # Extract from additional information
            additional_info = [a for a in annotations if 'Additional Information' in content[a['points'][0]['start']:a['points'][0]['end']]]
            if additional_info:
                skill_texts.append(content[additional_info[0]['points'][0]['start']:additional_info[0]['points'][0]['end']])

            # Combine all skill texts
            combined_skills = ' '.join(skill_texts)
            skills.append(self.preprocess_text(combined_skills))

        return skills

    def create_sample_job_descriptions(self):
        """Create sample job descriptions with required skills"""
        jobs = [
            {
                'title': 'Java Developer',
                'required_skills': ['java', 'spring', 'hibernate', 'sql', 'rest api'],
                'description': 'We are looking for a Java developer with experience in Spring framework and database technologies.'
            },
            {
                'title': 'Data Scientist',
                'required_skills': ['python', 'machine learning', 'pandas', 'numpy', 'data analysis'],
                'description': 'Seeking a data scientist with strong Python skills and machine learning experience.'
            },
            {
                'title': 'DevOps Engineer',
                'required_skills': ['docker', 'kubernetes', 'aws', 'ci/cd', 'terraform'],
                'description': 'Looking for a DevOps engineer with cloud and containerization experience.'
            },
            {
                'title': 'Web Developer',
                'required_skills': ['javascript', 'react', 'html', 'css', 'node.js'],
                'description': 'Front-end developer position requiring modern JavaScript frameworks.'
            },
            {
                'title': 'Database Administrator',
                'required_skills': ['sql', 'oracle', 'database management', 'performance tuning'],
                'description': 'DBA position requiring expertise in SQL and database optimization.'
            }
        ]
        self.job_descriptions = jobs
        return jobs

    def create_training_data(self, skills):
        """Create labeled training data based on skill matching"""
        X = []
        y = []

        for skill_text in skills:
            for job in self.job_descriptions:
                # Check if resume has at least 60% of required skills
                required_skills = set([self.preprocess_text(skill) for skill in job['required_skills']])
                resume_skills = set(skill_text.split())

                match_count = len(required_skills.intersection(resume_skills))
                match_percentage = match_count / len(required_skills)

                # Label as 1 (match) if at least 60% skills match
                label = 1 if match_percentage >= 0.6 else 0

                X.append(skill_text)
                y.append(label)

        return X, y

    def train_model(self, X, y):
        """Train the skill matching model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create pipeline with TF-IDF and classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        self.model = pipeline
        self.vectorizer = pipeline.named_steps['tfidf']

        return pipeline

    def predict_match(self, resume_text, job_title):
        """Predict if a resume matches a specific job"""
        if not self.model or not self.job_descriptions:
            raise ValueError("Model not trained or job descriptions not loaded")

        # Find the job
        job = next((j for j in self.job_descriptions if j['title'].lower() == job_title.lower()), None)
        if not job:
            raise ValueError(f"Job title '{job_title}' not found in known job descriptions")

        # Preprocess resume text
        processed_text = self.preprocess_text(resume_text)

        # Predict
        proba = self.model.predict_proba([processed_text])[0]
        prediction = self.model.predict([processed_text])[0]

        # Get matching skills
        required_skills = set([self.preprocess_text(skill) for skill in job['required_skills']])
        resume_skills = set(processed_text.split())
        matched_skills = required_skills.intersection(resume_skills)
        missing_skills = required_skills - resume_skills

        return {
            'job_title': job['title'],
            'match': bool(prediction),
            'confidence': max(proba),
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'required_skills': job['required_skills']
        }

    def process_resumes(self, json_data):
        """Process the JSON resume data and train model"""
        # Extract skills from resumes
        skills = self.extract_skills(json_data)

        # Create sample job descriptions
        self.create_sample_job_descriptions()

        # Create training data
        X, y = self.create_training_data(skills)

        # Train model
        self.train_model(X, y)

        return self

# Example usage with actual data
if __name__ == "__main__":
    # Initialize the matcher
    matcher = SkillMatcher()

    # Sample resume data (you would replace this with your actual JSON data)
    resume_data = [
        {
            "content": "Abhishek Jha\nApplication Development Associate - Accenture\n\nBengaluru, Karnataka - Email me on Indeed: indeed.com/r/Abhishek-Jha/10e7a8cb732bc43a\n\n• To work for an organization which provides me the opportunity to improve my skills\nand knowledge for my individual and company's growth in best possible ways.\n\nWilling to relocate to: Bangalore, Karnataka\n\nWORK EXPERIENCE\n\nApplication Development Associate\n\nAccenture -\n\nNovember 2017 to Present\n\nRole: Currently working on Chat-bot. Developing Backend Oracle PeopleSoft Queries\nfor the Bot which will be triggered based on given input. Also, Training the bot for different possible\nutterances (Both positive and negative), which will be given as\ninput by the user.\n\nEDUCATION\n\nB.E in Information science and engineering\n\nB.v.b college of engineering and technology -  Hubli, Karnataka\n\nAugust 2013 to June 2017\n\n12th in Mathematics\n\nWoodbine modern school\n\nApril 2011 to March 2013\n\n10th\n\nKendriya Vidyalaya\n\nApril 2001 to March 2011\n\nSKILLS\n\nC (Less than 1 year), Database (Less than 1 year), Database Management (Less than 1 year),\nDatabase Management System (Less than 1 year), Java (Less than 1 year)\n\nADDITIONAL INFORMATION\n\nTechnical Skills\n\nhttps://www.indeed.com/r/Abhishek-Jha/10e7a8cb732bc43a?isid=rex-download&ikw=download-top&co=IN\n\n\n• Programming language: C, C++, Java\n• Oracle PeopleSoft\n• Internet Of Things\n• Machine Learning\n• Database Management System\n• Computer Networks\n• Operating System worked on: Linux, Windows, Mac\n\nNon - Technical Skills\n\n• Honest and Hard-Working\n• Tolerant and Flexible to Different Situations\n• Polite and Calm\n• Team-Player",
            "annotation": [
                {"label": ["Skills"], "points": [{"start": 1295, "end": 1621, "text": "\n• Programming language: C, C++, Java\n• Oracle PeopleSoft\n• Internet Of Things\n• Machine Learning\n• Database Management System\n• Computer Networks\n• Operating System worked on: Linux, Windows, Mac\n\nNon - Technical Skills\n\n• Honest and Hard-Working\n• Tolerant and Flexible to Different Situations\n• Polite and Calm\n• Team-Player"}]},
                {"label": ["Skills"], "points": [{"start": 993, "end": 1153, "text": "C (Less than 1 year), Database (Less than 1 year), Database Management (Less than 1 year),\nDatabase Management System (Less than 1 year), Java (Less than 1 year)"}]}
            ]
        },
        {
            "content": "Avin Sharma\nSenior Associate Consultant - Infosys Limited\n\nHyderabad, Telangana - Email me on Indeed: indeed.com/r/Avin-Sharma/3ad8a8b57a172613\n\nWORK EXPERIENCE\n\nSenior Associate Consultant\n\nInfosys Limited -\n\nJuly 2015 to Present\n\nWorked on Presales activities preparing Proposals, RFP's, preparing presentations, budgets/\nquotations based\non client requirements.\n• Responsible for entire sales cycle from market research for prospective clients to final\nnegotiation & sales\nclosure.\n• Worked on mapping commercials to increase the profits and resources planning/ utilization.\n• Leading and coordinating a team of business analysts, developers and testers for execution\nof multiple projects.\n• Developed strategies and generated business for the firm by building corporate relationships\nwith client.\n\nSenior System Engineer\n\nInfosys Limited -\n\nAugust 2008 to April 2013\n\nWorked as a quality analyst for client; performed system testing for their customer web portal\nhaving modules\nlike Dashboard, Billing, Shop, Payments, Login etc.\n• Led a team of three system engineers; identified competency gap and conducted knowledge\nsharing sessions\nfor new team members.\n• Communicated with the client team members on frequent basis for understanding their issues\nand bringing\naction items to closure.\n• Independently handled the responsibility for designing manual test cases and executing them.\n• Worked on client's merger Integration application as a Developer which involved migration from\nlower framework\nto higher framework.\n• Responsible for migrating the part of application developed in dot net from lower version to\nhigher version.\n• Developed the internal portals on Dot Net platform for Infosys team members.\n• Completed the assigned tasks as per the timelines and ensured that the given tasks are\ncompleted with compliance to the benchmarks.\n\nhttps://www.indeed.com/r/Avin-Sharma/3ad8a8b57a172613?isid=rex-download&ikw=download-top&co=IN\n\n\n• Received STAR Certification for showcasing outstanding Soft Skills on 'Business & interpersonal\ncommunication\nfor Client delight.\n\nGuru Nanak Dev University\n\nAmritsar, Punjab -\n\nJuly 2004 to June 2008\n\nEDUCATION\n\nGreat Lakes Institute of Management -  Chennai, Tamil Nadu\n\nApril 2014 to April 2015\n\nSKILLS\n\nRequirement Analysis (Less than 1 year), Sales support (Less than 1 year), Test Planning (Less\nthan 1 year)\n\nADDITIONAL INFORMATION\n\nSkills\nBid management, Sales support, Requirement Analysis, Test Planning and Test execution",
            "annotation": [
                {"label": ["Skills"], "points": [{"start": 2394, "end": 2478, "text": "Bid management, Sales support, Requirement Analysis, Test Planning and Test execution"}]},
                {"label": ["Skills"], "points": [{"start": 2254, "end": 2360, "text": "Requirement Analysis (Less than 1 year), Sales support (Less than 1 year), Test Planning (Less\nthan 1 year)"}]}
            ]
        }
    ]

    # Process resumes and train model
    matcher.process_resumes(resume_data)

    # Example prediction
    sample_resume = """
    Experienced Java developer with 5 years of experience in Spring Boot, Hibernate, and REST APIs.
    Strong database skills with SQL and Oracle. Familiar with microservices architecture.
    """

    result = matcher.predict_match(sample_resume, "Java Developer")
    print("\nPrediction Result:")
    print(f"Job Title: {result['job_title']}")
    print(f"Match: {'Yes' if result['match'] else 'No'}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Matched Skills: {', '.join(result['matched_skills'])}")
    print(f"Missing Skills: {', '.join(result['missing_skills']) if result['missing_skills'] else 'None'}")
