document.getElementById('tourForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Fetch form data
    const formData = new FormData(event.target);

    // Create an object from form data
    const formDataObject = {};
    formData.forEach((value, key) => {
        formDataObject[key] = value;
    });

    // Make an API request using fetch
    fetch('/submit', {
        method: 'POST',
        body: new URLSearchParams(formData),
    })
    .then(response => response.json())
    .then(data => {
        // Redirect to results page with hotels data
        window.location.href = `/results?hotels=${encodeURIComponent(JSON.stringify(data))}`;
    })
    .catch(error => console.error('Error:', error));
});
