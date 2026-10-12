const form = document.querySelector('#prediction-form');
const prediction = document.querySelector('#prediction');
const message = document.querySelector('#message');
const randomButton = document.querySelector('#random-button');
const clearButton = document.querySelector('#clear-button');

const randomItem = (items) => items[Math.floor(Math.random() * items.length)];
const setValue = (name, value) => { document.querySelector(`[name="${name}"]`).value = value; };

randomButton.addEventListener('click', () => {
  setValue('service_intensity_weight', (0.7 + Math.random() * 2.4).toFixed(2));
  setValue('age_group', randomItem(['0 to 17', '18 to 29', '30 to 49', '50 to 69', '70 or Older']));
  setValue('gender', randomItem(['F', 'M']));
  setValue('ethnicity', randomItem(['Not Span/Hispanic', 'Spanish/Hispanic', 'Unknown']));
  setValue('drg', randomItem(['Medical', 'Surgical']));
  setValue('admission', randomItem(['Elective', 'Emergency', 'Trauma', 'Urgent']));
  setValue('illness', randomItem(['1', '2', '3', '4']));
  setValue('mortality', randomItem(['Extreme', 'Major', 'Minor', 'Moderate']));
  setValue('emergency', randomItem(['Y', 'N']));
  setValue('procedure', randomItem(['cardiovascular', 'endocrine', 'eye', 'nervous', 'others', 'physical']));
  setValue('mdc', randomItem(['1', '2', '3', '4', '5', 'Burn', 'Mental', 'Trauma']));
  setValue('facility', randomItem(['1245390855', '1326148669', '1346294717', '1427063270', '1679587679']));
  message.textContent = 'Sample values added. Review them or calculate now.';
  message.className = '';
});

clearButton.addEventListener('click', () => {
  form.reset();
  prediction.textContent = '--';
  message.textContent = 'Enter any available details, then calculate.';
  message.className = '';
});

function buildFeatures(data) {
  const features = {};
  if (data.service_intensity_weight !== '') features.service_intensity_weight = Number(data.service_intensity_weight);
  if (data.drg) features[`DRG_${data.drg}`] = 1;
  if (data.age_group) features[`age_group_${data.age_group}`] = 1;
  if (data.gender) features[`gender_${data.gender}`] = 1;
  if (data.ethnicity) features[`ethnicity_${data.ethnicity}`] = 1;
  if (data.admission) features[`type_of_admission_${data.admission}`] = 1;
  if (data.illness) features[`apr_severity_of_illness_code_${data.illness}`] = 1;
  if (data.mortality) features[`apr_risk_of_mortality_${data.mortality}`] = 1;
  if (data.emergency) features[`emergency_department_indicator_${data.emergency}`] = 1;
  if (data.procedure) features[`procedure_type_pocedure_${data.procedure}`] = 1;
  if (data.facility) features[`facility_name_${data.facility}`] = 1;
  if (data.mdc) features[data.mdc === '17' ? 'MDC__17' : `MDC_${data.mdc}`] = 1;
  return features;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  prediction.textContent = '--';
  const data = Object.fromEntries(new FormData(form).entries());
  const features = buildFeatures(data);

  if (Object.keys(features).length === 0) {
    message.textContent = 'Add at least one patient detail before calculating a forecast.';
    message.className = 'error';
    return;
  }

  message.textContent = 'Calculating forecast...';
  message.className = '';

  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Prediction failed');
    prediction.textContent = Number(result.predicted_length_of_stay_days).toFixed(1);
    message.textContent = 'Forecast generated from the available patient details.';
    message.className = 'success';
  } catch (error) {
    message.textContent = error.message;
    message.className = 'error';
  }
});
