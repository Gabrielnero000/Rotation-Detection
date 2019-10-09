from dataset import GetDataset
from model import GetModel

# Train params
epochs = 40
input_shape = (64, 64, 3)
num_classes = 4
batch_size = 128

# Get the dataset and the model
(train, val) = GetDataset(batch_size)
model = GetModel(input_shape, num_classes)

# Steps to use in fit_generator 
train_steps = train.n / train.batch_size
val_steps = val.n / val.batch_size

# Compile settings
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])

# The big deal
history = model.fit_generator(generator=train,
                              steps_per_epoch=train_steps,
                              validation_data=val,
                              validation_steps=val_steps,
                              epochs=epochs,
                              verbose=1)

# Save the model after train
model.save('model.h5')
