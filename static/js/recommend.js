const URL="https://api.unsplash.com/search/photos?page=1&query=marinedrive&client_id=mE5I6IJm60NvhRKgL6nVNdVvPHAowbmPj2mpZLIXwCE"

fetch(URL)
.then((res)=>{
    return res.json()
})
.then((data)=>{
    console.log(data)
})