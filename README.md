import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf

# -----------------------------
# 1. Define categories
# -----------------------------
crops = ['Maize', 'Wheat', 'Rice', 'Soybeans']
farming_types = ['Conventional', 'Organic', 'Hydroponic', 'Vertical']
districts = ['Bugesera', 'Gatsibo', 'Kayonza', 'Kirehe', 'Ngoma', 'Nyagatare',
             'Rwamagana', 'Kicukiro', 'Gasabo', 'Nyarugenge', 'Burera', 'Gakenke',
             'Gicumbi', 'Musanze', 'Rulindo', 'Gisagara', 'Huye', 'Kamonyi',
             'Muhanga', 'Nyamagabe', 'Nyanza', 'Nyaruguru', 'Ruhango', 'Karongi',
             'Ngororero', 'Nyabihu', 'Nyamasheke', 'Rubavu', 'Rusizi', 'Rutsiro']
fertilizers = ['Urea', 'DAP', 'MOP', 'NPK']
soil_types = ['Loamy', 'Sandy', 'Silty', 'Clay']

# -----------------------------
# 2. Generate synthetic dataset with patterns
# -----------------------------
np.random.seed(42)

data = {
    'Crop': np.random.choice(crops, 1000),
    'Farming Type': np.random.choice(farming_types, 1000),
    'District': np.random.choice(districts, 1000),
    'Fertilizer': np.random.choice(fertilizers, 1000),
    'Soil Type': np.random.choice(soil_types, 1000),
    'Area': np.random.uniform(1, 100, 1000)
}

df = pd.DataFrame(data)

# Create meaningful yield function
def generate_yield(row):
    base = 20

    # Crop effect
    crop_effect = {
        'Maize': 15, 'Wheat': 12, 'Rice': 18, 'Soybeans': 10
    }

    # Soil effect
    soil_effect = {
        'Loamy': 10, 'Sandy': -5, 'Silty': 5, 'Clay': 2
    }

    # Fertilizer effect
    fert_effect = {
        'NPK': 8, 'DAP': 6, 'Urea': 5, 'MOP': 4
    }

    # Farming type effect
    farming_effect = {
        'Conventional': 5, 'Organic': 3, 'Hydroponic': 12, 'Vertical': 10
    }

    yield_value = (
        base +
        crop_effect[row['Crop']] +
        soil_effect[row['Soil Type']] +
        fert_effect[row['Fertilizer']] +
        farming_effect[row['Farming Type']] +
        0.1 * row['Area'] +  # Area influence
        np.random.normal(0, 5)  # Noise
    )

    return yield_value

df['Yield'] = df.apply(generate_yield, axis=1)

# -----------------------------
# 3. Split features & target
# -----------------------------
X = df.drop('Yield', axis=1)
y = df['Yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Area']),
        ('cat', OneHotEncoder(sparse_output=False),
         ['Crop', 'Farming Type', 'District', 'Fertilizer', 'Soil Type'])
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# -----------------------------
# 5. Build model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['mean_absolute_error']
)

# -----------------------------
# 6. Train model
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 7. Evaluate
# -----------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: {mae:.2f}")

# -----------------------------
# 8. Predictions
# -----------------------------
predictions = model.predict(X_test)

predictions_df = pd.DataFrame({
    'Actual Yield': y_test.values,
    'Predicted Yield': predictions.flatten()
})

print("\nSample Predictions:")
print(predictions_df.head())