import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class IrisClassifier:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def load_data(self):
        """Load the Iris dataset"""
        iris = load_iris()
        self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.data['species'] = iris.target
        self.data['species_name'] = self.data['species'].apply(lambda x: iris.target_names[x])
        
        self.X = self.data[iris.feature_names]
        self.y = self.data['species']
        
        print("=" * 60)
        print("IRIS FLOWER DATASET LOADED")
        print("=" * 60)
        print(f"Dataset Shape: {self.data.shape}")
        print(f"\nFeatures: {list(iris.feature_names)}")
        print(f"Target Classes: {list(iris.target_names)}")
        print(f"\nFirst 5 rows:")
        print(self.data.head())
        
        return self.data
    
    def explore_data(self):
        """Explore and visualize the dataset"""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        print("\nDataset Statistics:")
        print(self.data.describe())
        
        print("\n\nClass Distribution:")
        print(self.data['species_name'].value_counts())
        
        # Create a 3x2 grid for plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Iris Dataset Exploration', fontsize=16, fontweight='bold')
        
        # 1. Distribution of species
        sns.countplot(x='species_name', data=self.data, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Iris Species')
        axes[0, 0].set_xlabel('Species')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Sepal length vs sepal width
        scatter1 = axes[0, 1].scatter(self.data['sepal length (cm)'], 
                                      self.data['sepal width (cm)'], 
                                      c=self.data['species'], cmap='viridis', s=50)
        axes[0, 1].set_title('Sepal Length vs Sepal Width')
        axes[0, 1].set_xlabel('Sepal Length (cm)')
        axes[0, 1].set_ylabel('Sepal Width (cm)')
        plt.colorbar(scatter1, ax=axes[0, 1], label='Species (0=setosa, 1=versicolor, 2=virginica)')
        
        # 3. Petal length vs petal width
        scatter2 = axes[1, 0].scatter(self.data['petal length (cm)'], 
                                      self.data['petal width (cm)'], 
                                      c=self.data['species'], cmap='plasma', s=50)
        axes[1, 0].set_title('Petal Length vs Petal Width')
        axes[1, 0].set_xlabel('Petal Length (cm)')
        axes[1, 0].set_ylabel('Petal Width (cm)')
        plt.colorbar(scatter2, ax=axes[1, 0], label='Species (0=setosa, 1=versicolor, 2=virginica)')
        
        # 4. Boxplot for sepal length by species
        sns.boxplot(x='species_name', y='sepal length (cm)', data=self.data, ax=axes[1, 1])
        axes[1, 1].set_title('Sepal Length by Species')
        axes[1, 1].set_xlabel('Species')
        axes[1, 1].set_ylabel('Sepal Length (cm)')
        
        # 5. Boxplot for petal length by species
        sns.boxplot(x='species_name', y='petal length (cm)', data=self.data, ax=axes[2, 0])
        axes[2, 0].set_title('Petal Length by Species')
        axes[2, 0].set_xlabel('Species')
        axes[2, 0].set_ylabel('Petal Length (cm)')
        
        # 6. Violin plot for petal width by species
        sns.violinplot(x='species_name', y='petal width (cm)', data=self.data, ax=axes[2, 1])
        axes[2, 1].set_title('Petal Width Distribution by Species')
        axes[2, 1].set_xlabel('Species')
        axes[2, 1].set_ylabel('Petal Width (cm)')
        
        plt.tight_layout()
        plt.show()
        
        # Create separate figure for feature distributions
        plt.figure(figsize=(12, 8))
        self.data.iloc[:, :4].hist(bins=15, edgecolor='black', layout=(2, 2))
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Correlation matrix
        plt.figure(figsize=(10, 8))
        numeric_cols = self.data.columns[:4]
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.show()
        
        # Pairplot
        print("\n" + "=" * 60)
        print("Pairplot of Features (Colored by Species)")
        print("=" * 60)
        sns.pairplot(self.data, hue='species_name', diag_kind='kde', 
                     palette='husl', height=2.5, plot_kws={'alpha': 0.7})
        plt.suptitle('Pairplot of Iris Features', y=1.02, fontsize=16, fontweight='bold')
        plt.show()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split and scale the data"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print(f"Number of features: {self.X_train.shape[1]}")
        print(f"Classes in training set: {np.unique(self.y_train)}")
        print(f"Classes in testing set: {np.unique(self.y_test)}")
        
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "=" * 60)
        print("TRAINING MULTIPLE CLASSIFICATION MODELS")
        print("=" * 60)
        
        # Define models with some tuned parameters
        models = {
            'Logistic Regression': LogisticRegression(max_iter=200, random_state=42, multi_class='ovr'),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(probability=True, random_state=42, kernel='rbf'),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train and evaluate each model
        results = []
        for name, model in models.items():
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            self.models[name] = model
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_train = model.predict(self.X_train_scaled)
            
            # Calculate accuracies
            test_accuracy = accuracy_score(self.y_test, y_pred)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Model': name,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'CV Score (Mean)': cv_mean,
                'CV Score (Std)': cv_std,
                'Overfitting Risk': train_accuracy - test_accuracy
            })
            
            print(f"\n{name}:")
            print(f"  Training Accuracy: {train_accuracy:.4f}")
            print(f"  Testing Accuracy: {test_accuracy:.4f}")
            print(f"  Cross-Validation Score: {cv_mean:.4f} (±{cv_std:.4f})")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        # Select best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "=" * 60)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print("=" * 60)
        
        return results_df
    
    def evaluate_best_model(self):
        """Evaluate the best performing model"""
        if self.best_model is None:
            print("No model has been trained yet!")
            return
        
        print(f"\n📊 Detailed Evaluation for {self.best_model_name}:")
        print("=" * 60)
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Classification report
        print("\n📈 CLASSIFICATION REPORT:")
        print("=" * 60)
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        print(classification_report(self.y_test, y_pred, target_names=target_names, digits=4))
        
        # Confusion matrix
        print("\n🔢 CONFUSION MATRIX:")
        print("=" * 60)
        cm = confusion_matrix(self.y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                            index=['Actual ' + name for name in target_names],
                            columns=['Predicted ' + name for name in target_names])
        print(cm_df)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'label': 'Number of Samples'})
        plt.title(f'Confusion Matrix - {self.best_model_name}\nAccuracy: {accuracy:.2%}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\n⚖️ FEATURE IMPORTANCE:")
            print("=" * 60)
            feature_importance = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(feature_importance.to_string(index=False))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
                           color='skyblue', edgecolor='black')
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Feature Importance - {self.best_model_name}', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', ha='left', va='center', fontweight='bold')
            
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()
        elif hasattr(self.best_model, 'coef_'):
            # For linear models like Logistic Regression
            print("\n📊 MODEL COEFFICIENTS:")
            print("=" * 60)
            if len(self.best_model.coef_.shape) > 1:
                # Multi-class case
                for i, class_coef in enumerate(self.best_model.coef_):
                    print(f"\nClass {i} ({target_names[i]}):")
                    for feature, coef in zip(self.X.columns, class_coef):
                        print(f"  {feature}: {coef:.4f}")
            else:
                # Binary case
                for feature, coef in zip(self.X.columns, self.best_model.coef_[0]):
                    print(f"{feature}: {coef:.4f}")
        
        return y_pred, y_pred_proba
    
    def make_prediction(self, sepal_length, sepal_width, petal_length, petal_width):
        """Make a prediction for new flower measurements"""
        if self.best_model is None:
            print("Please train the model first!")
            return None, None
        
        # Create input array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale the input
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.best_model.predict(input_scaled)[0]
        probabilities = self.best_model.predict_proba(input_scaled)[0]
        
        # Map prediction to species name
        species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        predicted_species = species_names[prediction]
        
        print("\n" + "=" * 60)
        print("🌺 IRIS FLOWER PREDICTION")
        print("=" * 60)
        print(f"\n📏 Input Measurements:")
        print(f"  Sepal Length: {sepal_length} cm")
        print(f"  Sepal Width: {sepal_width} cm")
        print(f"  Petal Length: {petal_length} cm")
        print(f"  Petal Width: {petal_width} cm")
        print(f"\n🔮 Prediction: {predicted_species}")
        print(f"\n📊 Prediction Probabilities:")
        for i, (species, prob) in enumerate(zip(species_names, probabilities)):
            prob_percent = prob * 100
            if i == prediction:
                print(f"  ✅ {species}: {prob_percent:.2f}% (PREDICTED)")
            else:
                print(f"  ○ {species}: {prob_percent:.2f}%")
        
        # Visualize the prediction
        self.visualize_prediction(input_data[0], predicted_species, probabilities)
        
        return predicted_species, probabilities
    
    def visualize_prediction(self, measurements, predicted_species, probabilities):
        """Visualize the prediction results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of probabilities
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Highlight predicted species
        bar_colors = colors.copy()
        predicted_idx = species_names.index(predicted_species.split('-')[1])
        
        bars = axes[0].bar(species_names, probabilities, color=bar_colors, edgecolor='black', linewidth=2)
        # Highlight the predicted bar
        bars[predicted_idx].set_edgecolor('gold')
        bars[predicted_idx].set_linewidth(3)
        
        axes[0].set_ylabel('Probability', fontsize=12)
        axes[0].set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Visualize the input measurements
        feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']
        
        bars2 = axes[1].bar(feature_names, measurements, color=colors, edgecolor='black')
        axes[1].set_ylabel('Measurement (cm)', fontsize=12)
        axes[1].set_title(f'Input Measurements\nPredicted: {predicted_species}', 
                         fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add measurement values on bars
        for bar, val in zip(bars2, measurements):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.1f} cm', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Iris Flower Classification Prediction', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 STARTING IRIS FLOWER CLASSIFICATION ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Prepare data
        self.prepare_data()
        
        # Step 4: Train models
        results_df = self.train_models()
        
        # Step 5: Evaluate best model
        self.evaluate_best_model()
        
        # Step 6: Example predictions
        print("\n" + "=" * 60)
        print("📝 EXAMPLE PREDICTIONS")
        print("=" * 60)
        
        # Example 1: Setosa (typically small petals)
        print("\n🌱 Example 1: Likely Iris-setosa")
        print("   (Typical characteristics: Small petals, wide sepals)")
        self.make_prediction(5.1, 3.5, 1.4, 0.2)
        
        # Example 2: Versicolor (medium petals)
        print("\n🌿 Example 2: Likely Iris-versicolor")
        print("   (Typical characteristics: Medium petals)")
        self.make_prediction(6.0, 2.7, 4.5, 1.5)
        
        # Example 3: Virginica (large petals)
        print("\n🌸 Example 3: Likely Iris-virginica")
        print("   (Typical characteristics: Large petals, narrow sepals)")
        self.make_prediction(7.7, 3.0, 6.1, 2.3)
        
        # Example 4: Borderline case (might be confusing)
        print("\n❓ Example 4: Borderline case")
        print("   (Characteristics between versicolor and virginica)")
        self.make_prediction(6.7, 3.0, 5.0, 1.7)
        
        return results_df

# Main execution
if __name__ == "__main__":
    try:
        # Initialize the classifier
        iris_classifier = IrisClassifier()
        
        # Run the complete analysis
        results = iris_classifier.run_complete_analysis()
        
        # Display model comparison
        print("\n" + "=" * 60)
        print("🏆 MODEL PERFORMANCE COMPARISON")
        print("=" * 60)
        print("\n" + results.to_string(index=False))
        
        # Create visualization of model comparison
        plt.figure(figsize=(12, 6))
        
        # Bar plot for test accuracy
        models = results['Model'].values
        test_acc = results['Test Accuracy'].values
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = plt.bar(models, test_acc, color=colors, edgecolor='black')
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title('Model Performance Comparison (Test Accuracy)', fontsize=14, fontweight='bold')
        plt.ylim([0.8, 1.02])
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, test_acc):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\n📋 Key Findings:")
        print("1. The Iris dataset contains 3 species with 4 features each")
        print("2. Petal measurements are more discriminative than sepal measurements")
        print("3. Most models achieve >90% accuracy on this well-separated dataset")
        print(f"4. Best performing model: {iris_classifier.best_model_name}")
        print("\n💡 Insights:")
        print("- Setosa is easily separable from the other two species")
        print("- Versicolor and Virginica have some overlap in measurements")
        print("- Petal length and width are the most important features")
        print("\n🎯 The model is now ready to classify iris flowers based on measurements!")
        
        # Interactive prediction option
        print("\n" + "=" * 60)
        print("🔮 TRY YOUR OWN PREDICTION")
        print("=" * 60)
        print("\nWant to try predicting your own iris flower measurements?")
        print("You can use the make_prediction() method:")
        print("\nExample:")
        print("  predicted_species = iris_classifier.make_prediction(5.8, 2.7, 5.1, 1.9)")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your input data and try again.")